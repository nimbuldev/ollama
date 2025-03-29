package llm

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
)

// This algorithm looks for a complete fit to determine if we need to unload other models
func PredictServerFit(allGpus discover.GpuInfoList, f *ggml.GGML, adapters, projectors []string, df *ggml.GGML, opts api.Options, numParallel int) (bool, uint64) {
	// Split up the GPUs by type and try them
	var estimatedVRAM, draftEstimatedVRAM uint64
	for _, gpus := range allGpus.ByLibrary() {
		var layerCount, draftLayerCount int
		estimate, draftEstimate := EstimateGPULayers(gpus, f, projectors, df, opts, numParallel)
		layerCount, estimatedVRAM = estimate.Layers, estimate.VRAMSize
				if isModelFitting(f, layerCount, opts.NumGPU) {
			// in case there is a draft model, also ensure that it fits
			if df == nil {
				return true, estimatedVRAM
			} else {
				draftLayerCount, draftEstimatedVRAM = draftEstimate.Layers, draftEstimate.VRAMSize
				if isModelFitting(df, draftLayerCount, opts.DraftNumGPU) {
					return true, estimatedVRAM + draftEstimatedVRAM
				}
			}
		}
	}
	return false, estimatedVRAM
}

func isModelFitting(f *ggml.GGML, layerCount int, numGPU int) bool {
	if numGPU < 0 {
		if layerCount > 0 && layerCount >= int(f.KV().BlockCount()+1) {
			return true
		}
	} else {
		if layerCount > 0 && layerCount >= numGPU {
			return true
		}
	}

	return false
}

type MemoryEstimate struct {
	// How many layers we predict we can load
	Layers int

	// The size of the graph which occupies the main GPU
	Graph uint64

	// How much VRAM will be allocated given the number of layers we predict
	VRAMSize uint64

	// The total size of the model if loaded into VRAM.  If all layers are loaded, VRAMSize == TotalSize
	TotalSize uint64

	// For multi-GPU scenarios, this provides the tensor split parameter
	TensorSplit string

	// For multi-GPU scenarios, this is the size in bytes per GPU
	GPUSizes []uint64

	// internal fields for logging purposes
	inferenceLibrary    string
	layersRequested     int
	layersModel         int
	availableList       []string
	kv                  uint64
	allocationsList     []string
	memoryWeights       uint64
	memoryLayerOutput   uint64
	graphFullOffload    uint64
	graphPartialOffload uint64
	projectorWeights, projectorGraph uint64
}

// Given a model, an optional draft model, and one or more GPU targets, predict how many layers and bytes we can load, and the total size
// The GPUs provided must all be the same Library
func EstimateGPULayers(gpus []discover.GpuInfo, f *ggml.GGML, projectors []string, df *ggml.GGML, opts api.Options, numParallel int) (MemoryEstimate, *MemoryEstimate) {
	var target, draft offloadLayoutRequirements

	overhead := envconfig.GpuOverhead()
	availableList := make([]string, len(gpus))
	for i, gpu := range gpus {
		availableList[i] = format.HumanBytes2(gpu.FreeMemory)
	}
	slog.Debug("evaluating", "library", gpus[0].Library, "gpu_count", len(gpus), "available", availableList)

	for _, projector := range projectors {
		weight, graph := projectorMemoryRequirements(projector)
		target.projectorWeights += weight
		target.projectorGraph += graph

		// multimodal models require at least 2048 context
		opts.NumCtx = max(opts.NumCtx, 2048)
	}

	fa := envconfig.FlashAttention() &&
		discover.GetGPUInfo().FlashAttentionSupported() &&
		f.SupportsFlashAttention()

	calculateLayersSizes(gpus, f, opts.NumCtx, opts.NumBatch, fa, &target, numParallel)

	// Output layer handled at the end if we have space
	gpuZeroOverhead := target.projectorWeights + target.projectorGraph

	// Reduce set of GPUs to only those that have sufficient space to fit overhead and at least one layer
	target.layerCounts = make([]int, len(gpus))
	draft.layerCounts = make([]int, len(gpus))
	gpuAllocations := make([]uint64, len(gpus))
	var gpusWithSpace []gs

	gpusWithSpace = allocateInitialLayersToGPUs(gpus, overhead, gpuZeroOverhead, &target, gpuAllocations)
	allocateLayersToGPUs(gpusWithSpace, f, opts.NumGPU, overhead, &target, gpuAllocations)
	allocateGraphToGPUs(gpus, &target, gpuAllocations)
	targetEstimate := newEstimate(gpus, f, opts.NumGPU, gpuAllocations, &target)

	if df != nil {
		var draftEstimate MemoryEstimate
		// draft models do not use flash attention or KV cache quantizations
		calculateLayersSizes(gpus, df, opts.DraftNumCtx, opts.DraftNumCtx, false, &draft, numParallel)
		gpusWithSpace = allocateInitialLayersToGPUs(gpus, overhead, 0, &draft, gpuAllocations)
		allocateLayersToGPUs(gpusWithSpace, df, opts.DraftNumGPU, overhead, &draft, gpuAllocations)
		allocateGraphToGPUs(gpus, &draft, gpuAllocations)
		for i := range gpuAllocations {
			gpuAllocations[i] -= targetEstimate.GPUSizes[i]
		}
		draftEstimate = newEstimate(gpus, df, opts.DraftNumGPU, gpuAllocations, &draft)
		return targetEstimate, &draftEstimate
	}
	return targetEstimate, nil
}


func (m MemoryEstimate) LogValue() slog.Value {
	attrs := []slog.Attr{
		slog.String("library", m.inferenceLibrary),
		slog.Group(
			"layers",
			// requested number of layers to offload
			"requested", m.layersRequested,
			// The number of layers the model has (including output)
			"model", m.layersModel,
			// estimated number of layers that can be offloaded
			"offload", m.Layers,
			// multi-gpu split for tensors
			"split", m.TensorSplit,
		),
		slog.Group(
			"memory",
			// memory available by GPU for offloading
			"available", m.availableList,
			"gpu_overhead", format.HumanBytes2(envconfig.GpuOverhead()),
			slog.Group(
				"required",
				// memory required for full offloading
				"full", format.HumanBytes2(m.TotalSize),
				// memory required to offload layers.estimate layers
				"partial", format.HumanBytes2(m.VRAMSize),
				// memory of KV cache
				"kv", format.HumanBytes2(m.kv),
				// Allocations across the GPUs
				"allocations", m.allocationsList,
			),
			slog.Group(
				"weights",
				// memory of the weights
				"total", format.HumanBytes2(m.memoryWeights+m.memoryLayerOutput),
				// memory of repeating layers
				"repeating", format.HumanBytes2(m.memoryWeights),
				// memory of non-repeating layers
				"nonrepeating", format.HumanBytes2(m.memoryLayerOutput),
			),
			slog.Group(
				"graph",
				// memory of graph when fully offloaded
				"full", format.HumanBytes2(m.graphFullOffload),
				// memory of graph when not fully offloaded
				"partial", format.HumanBytes2(m.graphPartialOffload),
			),
		),
	}

	if m.projectorWeights > 0 {
		attrs = append(attrs, slog.Group(
			"projector",
			"weights", format.HumanBytes2(m.projectorWeights),
			"graph", format.HumanBytes2(m.projectorGraph),
		))
	}

	return slog.GroupValue(attrs...)
}

func projectorMemoryRequirements(filename string) (weights, graphSize uint64) {
	file, err := os.Open(filename)
	if err != nil {
		return 0, 0
	}
	defer file.Close()

	ggml, _, err := ggml.Decode(file, 0)
	if err != nil {
		return 0, 0
	}

	for _, layer := range ggml.Tensors().GroupLayers() {
		weights += layer.Size()
	}

	switch arch := ggml.KV().Architecture(); arch {
	case "mllama":
		kv := func(n string) uint64 {
			if v, ok := ggml.KV()[arch+".vision."+n].(uint32); ok {
				return uint64(v)
			}

			return 0
		}

		imageSize := kv("image_size")

		maxNumTiles := kv("max_num_tiles")
		embeddingLength := kv("embedding_length")
		headCount := kv("attention.head_count")

		numPatches := (imageSize / kv("patch_size")) * (imageSize / kv("patch_size"))
		if _, ok := ggml.Tensors().GroupLayers()["v"]["class_embd"]; ok {
			numPatches++
		}

		numPaddedPatches := numPatches + 8 - (numPatches%8)%8

		graphSize = 4 * (8 +
			imageSize*imageSize*kv("num_channels")*maxNumTiles +
			embeddingLength*numPatches*maxNumTiles +
			9*embeddingLength*numPaddedPatches*maxNumTiles +
			numPaddedPatches*maxNumTiles*numPaddedPatches*maxNumTiles*headCount)
	}

	return weights, graphSize
}

type offloadLayoutRequirements struct {
	
	graphPartialOffload uint64

	// Graph size when all layers are offloaded, applies to all GPUs
	graphFullOffload uint64

	// Final graph offload once we know full or partial
	graphOffload uint64

	// Projectors loaded into GPU0 only
	projectorWeights uint64
	projectorGraph uint64

	// Conditional output size on GPU 0
	memoryLayerOutput uint64

	// The sizes of a layer
	layerSize uint64

	// The sum of all the layer sizes (just for logging)
	memoryWeights uint64

	// True if all the layers are loaded
	fullyLoaded bool

	// Overflow that didn't fit into the GPU
	overflow uint64

	kv []uint64
	kvTotal uint64

	layerCount int

	layerCounts []int
}

type gs struct {
	i int
	g *discover.GpuInfo
}

func calculateLayersSizes(gpus []discover.GpuInfo, f *ggml.GGML, numCtx int, numBatch int, fa bool, olrs *offloadLayoutRequirements, numParallel int) {
	if olrs.projectorWeights == 0 && olrs.projectorGraph == 0 {
		olrs.projectorWeights, olrs.projectorGraph = f.VisionGraphSize()
	}
	
	layers := f.Tensors().GroupLayers()
	// add one layer worth of memory as a buffer
	if blk0, ok := layers["blk.0"]; ok {
		olrs.layerSize = blk0.Size()
	} else {
		slog.Warn("model missing blk.0 layer size")
	}

	var kvct string
	if fa {
		requested := strings.ToLower(envconfig.KvCacheType())
		if requested != "" && f.SupportsKVCacheType(requested) {
			kvct = requested
		}
	}

	olrs.kv, olrs.graphPartialOffload, olrs.graphFullOffload = f.GraphSize(uint64(numCtx), uint64(min(numCtx, numBatch)), numParallel, kvct)


	if len(olrs.kv) > 0 {
		olrs.layerSize += olrs.kv[0]
	}

	for _, kvLayer := range olrs.kv {
		olrs.kvTotal += kvLayer
	}

	if olrs.graphPartialOffload == 0 {
		olrs.graphPartialOffload = f.KV().GQA() * olrs.kvTotal / 6
	}
	if olrs.graphFullOffload == 0 {
		olrs.graphFullOffload = olrs.graphPartialOffload
	}

	// on metal there's no partial offload overhead
	if gpus[0].Library == "metal" {
		olrs.graphPartialOffload = olrs.graphFullOffload
	} else if len(gpus) > 1 {
		// multigpu should always use the partial graph size
		olrs.graphFullOffload = olrs.graphPartialOffload
	}

	if layer, ok := layers["output_norm"]; ok {
		olrs.memoryLayerOutput += layer.Size()
	}
	if layer, ok := layers["output"]; ok {
		olrs.memoryLayerOutput += layer.Size()
	} else if layer, ok := layers["token_embd"]; ok {
		olrs.memoryLayerOutput += layer.Size()
	}
}

func allocateInitialLayersToGPUs(gpus []discover.GpuInfo, overhead uint64, gpuZeroOverhead uint64, olrs *offloadLayoutRequirements, gpuAllocations []uint64) (gpusWithSpace []gs) {
	gpusWithSpace = []gs{}
	for i := range gpus {
		var gzo uint64
		if len(gpusWithSpace) == 0 {
			gzo = gpuZeroOverhead
		}
		used := gpuAllocations[i]
		// Only include GPUs that can fit the graph, gpu minimum, the layer buffer and at least one more layer
		if gpus[i].FreeMemory < used+overhead+gzo+max(olrs.graphPartialOffload, olrs.graphFullOffload)+gpus[i].MinimumMemory+2*olrs.layerSize {
			slog.Debug("gpu has too little memory to allocate any layers",
				"id", gpus[i].ID,
				"library", gpus[i].Library,
				"variant", gpus[i].Variant,
				"compute", gpus[i].Compute,
				"driver", fmt.Sprintf("%d.%d", gpus[i].DriverMajor, gpus[i].DriverMinor),
				"name", gpus[i].Name,
				"total", format.HumanBytes2(gpus[i].TotalMemory),
				"available", format.HumanBytes2(gpus[i].FreeMemory),
				"minimum_memory", gpus[i].MinimumMemory,
				"layer_size", format.HumanBytes2(olrs.layerSize),
				"gpu_zer_overhead", format.HumanBytes2(gzo),
				"partial_offload", format.HumanBytes2(olrs.graphPartialOffload),
				"full_offload", format.HumanBytes2(olrs.graphFullOffload),
			)
			continue
		}
		gpusWithSpace = append(gpusWithSpace, gs{i, &gpus[i]})
		gpuAllocations[i] += gpus[i].MinimumMemory + olrs.layerSize // We hold off on graph until we know partial vs. full
	}

	var gpuZeroID int
	if len(gpusWithSpace) > 0 {
		gpuZeroID = gpusWithSpace[0].i
		gpuAllocations[gpuZeroID] += gpuZeroOverhead
	}
	return gpusWithSpace
}

func allocateLayersToGPUs(gpusWithSpace []gs, f *ggml.GGML, numGPU int, overhead uint64, olrs *offloadLayoutRequirements, gpuAllocations []uint64) {
	layers := f.Tensors().GroupLayers()

	// For all the layers, find where they can fit on the GPU(s)
	for i := range int(f.KV().BlockCount()) {
		// Some models have inconsistent layer sizes
		if blk, ok := layers[fmt.Sprintf("blk.%d", i)]; ok {
			olrs.layerSize = blk.Size()
			olrs.layerSize += olrs.kv[i]
			olrs.memoryWeights += blk.Size()
		}

		if numGPU >= 0 && olrs.layerCount >= numGPU {
			// Stop allocating on GPU(s) once we hit the users target NumGPU
			continue
		}

		// distribute the layers across the GPU(s) that have space
		for j := len(gpusWithSpace); j > 0; j-- {
			g := gpusWithSpace[i%j]
			used := gpuAllocations[g.i] + max(olrs.graphPartialOffload, olrs.graphFullOffload)
			if g.g.FreeMemory > overhead+used+olrs.layerSize {
				gpuAllocations[g.i] += olrs.layerSize
				olrs.layerCounts[g.i]++
				olrs.layerCount++
				break
			} else {
				gpusWithSpace = append(gpusWithSpace[:i%j], gpusWithSpace[i%j+1:]...)
			}
		}
	}
	if olrs.layerCount >= int(f.KV().BlockCount()) {
		olrs.fullyLoaded = true
	} else {
		for i := olrs.layerCount; i < int(f.KV().BlockCount()); i++ {
			olrs.overflow += olrs.layerSize
		}
	}

	// Determine if we need to consider output then find where it fits
	if olrs.memoryLayerOutput > 0 && (numGPU < 0 || olrs.layerCount < numGPU) {
		for j := len(gpusWithSpace); j > 0; j-- {
			g := gpusWithSpace[olrs.layerCount%j]
			used := gpuAllocations[g.i] + max(olrs.graphPartialOffload, olrs.graphFullOffload)
			if g.g.FreeMemory > overhead+used+olrs.memoryLayerOutput {
				gpuAllocations[g.i] += olrs.memoryLayerOutput
				olrs.layerCounts[g.i]++
				olrs.layerCount++
				break
			}
		}

		if olrs.layerCount < int(f.KV().BlockCount())+1 {
			olrs.fullyLoaded = false
			olrs.overflow += olrs.memoryLayerOutput
		}
	}
}

func allocateGraphToGPUs(gpus []discover.GpuInfo, olrs *offloadLayoutRequirements, gpuAllocations []uint64) {
	// Add the applicable (full or partial) graph allocations
	for i := range gpus {
		if olrs.layerCounts[i] <= 0 {
			continue
		}
		if olrs.fullyLoaded {
			gpuAllocations[i] += olrs.graphFullOffload
		} else {
			gpuAllocations[i] += olrs.graphPartialOffload
		}
	}
	if olrs.fullyLoaded {
		olrs.graphOffload = olrs.graphFullOffload
	} else {
		olrs.graphOffload = olrs.graphPartialOffload
	}
}


func newEstimate(gpus []discover.GpuInfo, f *ggml.GGML, numGPU int, gpuAllocations []uint64, olrs *offloadLayoutRequirements) MemoryEstimate {
	
	availableList := make([]string, len(gpus))
	for i, gpu := range gpus {
		availableList[i] = format.HumanBytes2(gpu.FreeMemory)
	}

	// Summaries for the log
	var memoryRequiredPartial, memoryRequiredTotal uint64
	for i := range gpuAllocations {
		memoryRequiredPartial += gpuAllocations[i]
	}
	memoryRequiredTotal = memoryRequiredPartial + olrs.overflow

	tensorSplit := ""
	if len(gpus) > 1 {
		splits := make([]string, len(gpus))
		for i, count := range olrs.layerCounts {
			splits[i] = strconv.Itoa(count)
		}
		tensorSplit = strings.Join(splits, ",")
	}
	allocationsList := []string{}
	for _, a := range gpuAllocations {
		allocationsList = append(allocationsList, format.HumanBytes2(a))
	}

	estimate := MemoryEstimate{
		TotalSize: memoryRequiredTotal,
		Layers:    0,
		Graph:     0,
		VRAMSize:  0,
		GPUSizes:  []uint64{},

		inferenceLibrary:    gpus[0].Library,
		layersRequested:     numGPU,
		layersModel:         int(f.KV().BlockCount()) + 1,
		availableList:       availableList,
		kv:                  olrs.kvTotal,
		allocationsList:     allocationsList,
		memoryWeights:       olrs.memoryWeights,
		memoryLayerOutput:   olrs.memoryLayerOutput,
		graphFullOffload:    olrs.graphFullOffload,
		graphPartialOffload: olrs.graphPartialOffload,
		projectorWeights:    olrs.projectorWeights,
		projectorGraph:      olrs.projectorGraph,
	}

	if gpus[0].Library == "cpu" {
		return estimate
	}
	if olrs.layerCount == 0 {
		slog.Debug("insufficient VRAM to load any model layers")
		return estimate
	}
	estimate.Layers = olrs.layerCount
	estimate.Graph = olrs.graphOffload
	estimate.VRAMSize = memoryRequiredPartial
	estimate.TotalSize = memoryRequiredTotal
	estimate.TensorSplit = tensorSplit
	estimate.GPUSizes = gpuAllocations
	return estimate
}
