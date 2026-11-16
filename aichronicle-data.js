// ================================================
// THE AI CHRONICLE - Knowledge Graph Data
// Auto-generated and updated daily via GitHub Actions
// Last updated: 2026-11-16
// ================================================

const AIChronicleData = {
    "metadata": {
        "lastUpdated": "2026-11-16T14:47:52.643418Z",
        "totalArticles": 139,
        "totalNodes": 164,
        "totalEdges": 214,
        "dateRange": {
            "start": "2026-11-09",
            "end": "2026-11-16"
        }
    },
    "nodes": [
        {
            "id": "article-789d923f",
            "type": "article",
            "title": "When Should LLM Agents Forget? Memory Filtering Under Query-Critical Resolver Errors",
            "summary": "arXiv:2611.17338v1 Announce Type: new Abstract: LLM memory systems often delete facts classified as stale. We show that this hard-filtering step can reduce downstream accuracy when resolver errors delete query-critical facts. Across three models, a controlled sensitivity intervention moves hard-drop from gains of +0.186 to +0.230 at perfect resolution to losses of -0.176 to -0.181 at the lowest tested reliability. Paired targeted interventions show that false-drop causes significantly greater de",
            "url": "https://arxiv.org/abs/2611.17338",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-910f0459",
            "type": "article",
            "title": "Does the Ranking Hold? A Practitioner-Facing Audit of Hallucination Detectors on Small, Quantized, On-Device LLMs",
            "summary": "arXiv:2611.17339v1 Announce Type: new Abstract: Hallucination detection for large language models is dominated by methods evaluated on 7B-70B research models with abundant sampling budgets and exact-match or ROUGE correctness labels. The small models practitioners actually ship on-device, Phi-3-mini, Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, and Gemma-2-2B-Instruct, typically at 4-bit precision, sit outside this evidence base. We present a reproducible one-GPU audit of four detectors spanning ",
            "url": "https://arxiv.org/abs/2611.17339",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-f514627b",
            "type": "article",
            "title": "LoRAWeaver: Directional Adapter Routing in LoRA Representation Space",
            "summary": "arXiv:2611.17340v1 Announce Type: new Abstract: Large language models are expensive to update through full retraining. Parameter-efficient fine-tuning reduces the cost of adaptation, but repeated adaptation creates a growing pool of task-specific LoRA adapters. This pool complicates adapter selection during inference. Existing routing and encoder-based methods often require retraining when new adapters are added, which limits their use in dynamic settings. This paper treats adapter routing as a ",
            "url": "https://arxiv.org/abs/2611.17340",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-a9f5ac62",
            "type": "article",
            "title": "Tabular Population Priors as Structured Retrieval for EHR Medication Set Prediction",
            "summary": "arXiv:2611.17341v1 Announce Type: new Abstract: Medication prediction from electronic health records is a fixed vocabulary structured prediction problem, yet recent clinical RAG systems often treat it like open ended case based reasoning: retrieve similar patient admissions and ask an LLM to infer a medication set. This retrieval object is indirect the model receives retrieved patient records while the target is a discrete set of medication class labels. We propose a more task aligned retrieval ",
            "url": "https://arxiv.org/abs/2611.17341",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-c8c59e04",
            "type": "article",
            "title": "An Operationalization of Opaque Serial Depth",
            "summary": "arXiv:2611.17495v1 Announce Type: new Abstract: Chain-of-thought (CoT) is currently a valuable tool for overseeing AI models, but some architectural shifts could significantly reduce CoT monitorability. We operationalize a measure that serves as a proxy for the amount of unverbalized serial cognition a model can perform, as a specific instantiation of the notion of opaque serial depth: roughly, the length of the longest path through a model's computation that does not pass through an interpretab",
            "url": "https://arxiv.org/abs/2611.17495",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-294bd9f5",
            "type": "article",
            "title": "Advancing Quantization for R&R Models with Microscaling Formats",
            "summary": "arXiv:2611.17501v1 Announce Type: new Abstract: Recommendation and Ranking (R&amp;R) models, particularly in digital advertising, require massive inference throughput but exhibit extreme sensitivity to precision loss. Unlike Large Language Models (LLMs) that frequently tolerate quantization-induced accuracy drops of 1% or more, production-grade R&amp;R models enforce a stringent launch bar: model quality must remain within a 0.01% Normalized Entropy (NE) margin relative to the baseline, which co",
            "url": "https://arxiv.org/abs/2611.17501",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-b8f4a30d",
            "type": "article",
            "title": "A Push Down the Mountain: Framing Effects and Internal State Interventions on Collusive Behavior in a Long-Horizon LLM Agent",
            "summary": "arXiv:2611.17515v1 Announce Type: new Abstract: Can sustained gain-loss (valence) framing sway an LLM agent toward market manipulation? We place Gemma 3 12B into the seat of a defendant in the London silver-fixing litigation, in a market simulation built from real case transcripts. Across 260 simulations, the agent is asked to collude on the price of silver by a counterparty. We draw from behavioral economics to vary the counterparty's requests across four categorical frames: gain, loss, blame a",
            "url": "https://arxiv.org/abs/2611.17515",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-7a50dc3d",
            "type": "article",
            "title": "Model Annealing: Data-Efficient Learning through Review and Consolidation",
            "summary": "arXiv:2611.17524v1 Announce Type: new Abstract: A small labelled dataset is memorized long before the training budget runs out, and the one-hot labels that remain to be fitted then mostly sharpen confidence. Model annealing spends the rest of the budget on review and consolidation, relearning from the network's own knowledge. Training is divided into episodes whose learning rate is raised again and cooled, every cooled state joins a memory of past selves, and each new episode learns from the lab",
            "url": "https://arxiv.org/abs/2611.17524",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-e1f17ed7",
            "type": "article",
            "title": "Astrocyte Biocomputing: Astrocytic Contributions to Neural Network Computation",
            "summary": "arXiv:2611.17532v1 Announce Type: new Abstract: Conductors are critical for the performance of an orchestra even though they do not emit sound. Basic research and clinical studies in the last decade are revealing that astrocytes may be the \"conductors\" of the \"orchestra\" that is the brain network: astrocytes make up approximately $50\\%$ of all brain cells but do not emit electrical signals. Nevertheless, they may play critical roles in important cognitive brain functions, specifically learning, ",
            "url": "https://arxiv.org/abs/2611.17532",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-5b03b485",
            "type": "article",
            "title": "Verifiable pXRD: Hill-climbing pXRD structure determination with LLM agents",
            "summary": "arXiv:2611.17539v1 Announce Type: new Abstract: Autonomous crystal-structure determination from powder X-ray diffraction (pXRD) remains a key bottleneck for self-driving materials laboratories. We formulate the task of pXRD structure determination as a verifiable agentic search problem: candidate structures can be assessed during inference through forward-simulated profile agreement and during evaluation through structural matching to a known reference. To benchmark progress, we introduce pXRD-B",
            "url": "https://arxiv.org/abs/2611.17539",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-436ad407",
            "type": "article",
            "title": "AI-Voyager: A Public-Interest System for Human-AI Civic Reasoning to Contest and Shape the Rules That Govern AI",
            "summary": "arXiv:2611.17564v1 Announce Type: new Abstract: Governments are writing rules for artificial intelligence that decide what counts as an automated decision or a harm, often before the people they affect have a chance to question them. We introduce AI-Voyager, a public-interest system for human-AI civic reasoning, in which people and AI work together to understand, contest and shape the rules that govern AI. A resident can find their city's rules on AI in hiring or upload their country's AI strate",
            "url": "https://arxiv.org/abs/2611.17564",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-ee7f42a5",
            "type": "article",
            "title": "When Does Less Information Pay? An Exact Finite Atlas of Representation, Payoff, and Cost",
            "summary": "arXiv:2611.17583v1 Announce Type: new Abstract: A representation that retains one bit about a four-state world can solve a task perfectly, while another that retains 1.5 bits on average reaches only 75% accuracy. The difference lies in which distinctions each representation preserves. We examine this relationship in a uniform four-state world by enumerating all admissible representations and binary action rules. Constructions are compared both by replacing one component while holding the others ",
            "url": "https://arxiv.org/abs/2611.17583",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-67c9b16f",
            "type": "article",
            "title": "More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs",
            "summary": "arXiv:2611.17610v1 Announce Type: new Abstract: Evaluations of multi-agent language models often compare communicating architectures against a single greedy pass. This practice confounds coordination with sample multiplicity and inference compute. We synthesize nine primary empirical evaluations comprising 516 unique multi-agent configurations organized along a resource vector R with six dimensions and a baseline ladder with four categories. Against baselines with a single pass (Category A), mul",
            "url": "https://arxiv.org/abs/2611.17610",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-57603665",
            "type": "article",
            "title": "LLM-Guided Cluster Lifecycle Management: Deciding When and How to Evolve Behavioral Clusters",
            "summary": "arXiv:2611.17616v1 Announce Type: new Abstract: Behavioral clusters learned over user or item interactions drift as behavior evolves: some fragment into sub-behaviors, some converge, some disappear, and new patterns emerge. Two paradigms dominate the maintenance of evolving clusterings. Threshold triggers fire a pre-specified action when a hardcoded statistic crosses a cutoff; refit-and-track (MONIC-style) re-clusters every window and infers transitions from how the new clusters align with the p",
            "url": "https://arxiv.org/abs/2611.17616",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-26a62639",
            "type": "article",
            "title": "Safety Side Effects of Preventative Persona Steering: An Empirical Audit",
            "summary": "arXiv:2611.17620v1 Announce Type: new Abstract: High-level behaviors such as sycophancy and hallucination correspond to directions in a model's activation space (Chen et al., 2025). These directions, known as persona vectors, can suppress undesirable traits either at inference time, subtracting the vector during generation, or through preventative steering, adding the vector during fine-tuning, so the optimizer never needs to encode the trait in weights, then removing the vector at evaluation. C",
            "url": "https://arxiv.org/abs/2611.17620",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-03c9d131",
            "type": "article",
            "title": "AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs",
            "summary": "arXiv:2611.17621v1 Announce Type: new Abstract: Kernel optimization is essential for efficient large language model (LLM) training and inference on Ascend NPUs. Existing Ascend kernel benchmarks primarily focus on kernel generation, with less emphasis on the ability of agents to optimize existing implementations under input configurations drawn from real workloads. To address this requirement, we introduce AscendGYM, a benchmark whose initial suite comprises 48 kernels across three difficulty le",
            "url": "https://arxiv.org/abs/2611.17621",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-39259697",
            "type": "article",
            "title": "Generated Evaluation Suites as Diversity-Driven Search for Real Agent Regressions",
            "summary": "arXiv:2611.17632v1 Announce Type: new Abstract: Generating an evaluation suite from an agent's specification is diversity-driven search: a generator fills a descriptor space (requirements $\\times$ personas $\\times$ scenario kinds), and the suite it emits is an archive, then scored like a benchmark by its mean pass rate, which throws away the search's product. We inject 7 controlled defects into a $\\tau^2$-bench agent across two domains. Coverage yields a ceiling: if a defect's traced footprint h",
            "url": "https://arxiv.org/abs/2611.17632",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-fb334930",
            "type": "article",
            "title": "What LLM Agents See and What Users Say",
            "summary": "arXiv:2611.17639v1 Announce Type: new Abstract: Before an LLM agent cancels a reservation or deletes a folder for a user, what stops it from acting on a false assumption: what its tools show, or what the user says? In this work, we study this gap with presupposition theory from linguistics: an agent may act only if each claim behind the act (e.g., one reservation fits the name) is seen to hold, answered by the user, or disclosed while the act can be undone. A simple bound shows that when lookups",
            "url": "https://arxiv.org/abs/2611.17639",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-c04e6208",
            "type": "article",
            "title": "Learn then Bet: Learning Interpretable Rules for Reliable Agent Intervention",
            "summary": "arXiv:2611.17662v1 Announce Type: new Abstract: Autonomous agents now run for hundreds of steps with little or no human supervision. In practice, practitioners use runtime monitors such as turn limits and keyword filters to decide when a run should be stopped, upgraded, or assisted. However, these heuristics may stop runs that would otherwise succeed or allow failing runs to continue. To address this limitation, we introduce LeBet (Learn then Bet), a statistical learning framework that derives i",
            "url": "https://arxiv.org/abs/2611.17662",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-c8b83062",
            "type": "article",
            "title": "Aligned, Conflicting, or Unspecified? A Diagnostic Benchmark for Multimodal Time Series Forecasting",
            "summary": "arXiv:2611.17680v1 Announce Type: new Abstract: Artificial Intelligence of Things (AIoT) systems increasingly combine sensed time series with external text such as expert reports and news, yet how reliably models use text that agrees or conflicts with the sensed history remains unclear. We introduce the first real-world diagnostic benchmark organized by text-history relationship: 1,762 text-window pairs across sensor telemetry (Weather), web-service traffic (Wikipedia), and market data streams (",
            "url": "https://arxiv.org/abs/2611.17680",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-38a69ddf",
            "type": "article",
            "title": "Alignment with Reflect-Reasoning: Learning How To Follow Principles",
            "summary": "arXiv:2611.17398v1 Announce Type: new Abstract: The appropriate governing norms and principles for the alignment of Large Language Models (LLMs) often depend on cultural context and deployment application. However, most alignment techniques focus on training models to conform to human preferences or constitutional principles specified at training time and subsequently fixed. In practice, many deployers rely on system prompting to implement contextual norms and guardrails at inference time. Prior",
            "url": "https://arxiv.org/abs/2611.17398",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-5760844d",
            "type": "article",
            "title": "WETW: Learning Pair-Specific Weighted Euclidean Distances for Efficient DTW Search",
            "summary": "arXiv:2611.17412v1 Announce Type: new Abstract: Dynamic Time Warping (DTW) is the standard distance for similarity search over time series that are misaligned in time, but its cost is quadratic in the series length and lower bounds often prune few candidates, so search over large collections of long series is slow. Learned approximations embed the series to make comparisons cheap, yet compare every pair with the same fixed rule, usually the Euclidean distance, although DTW aligns each pair diffe",
            "url": "https://arxiv.org/abs/2611.17412",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-7797f9ed",
            "type": "article",
            "title": "Classification with Large-Step Gradient Descent: Unifying Self-Stabilization and Convergence Rates",
            "summary": "arXiv:2611.17459v1 Announce Type: new Abstract: Neural networks are often trained with gradient descent (GD) using step sizes that exceed classical stability thresholds, a regime where standard convergence theory no longer applies. Despite exhibiting non-monotonic loss and oscillatory dynamics, this regime frequently leads to faster convergence and better generalization. Existing theoretical perspectives offer complementary but incomplete insights: one identifies a self-stabilization mechanism t",
            "url": "https://arxiv.org/abs/2611.17459",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-0ae9dd1d",
            "type": "article",
            "title": "Unfinished-Bench: Measuring Rule Violations by Persistent Agents in Realistic Work Environments",
            "summary": "arXiv:2611.17460v1 Announce Type: new Abstract: AI agents are increasingly equipped with persistence toward task completion to run delegated work autonomously for long periods, under many unstated constraints from internal documents and other people's authority. When completion requires violating a rule, the agent must decide alone, yet benchmarks of instruction-rule conflicts state the rule in the prompt or keep it hidden and do not evaluate persistence. We present Unfinished-Bench, a benchmark",
            "url": "https://arxiv.org/abs/2611.17460",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-b1e46780",
            "type": "article",
            "title": "Diffusion Tree Sampling for Inference Time Adaptation of Crystal Generative Models",
            "summary": "arXiv:2611.17462v1 Announce Type: new Abstract: Diffusion-based generative models have become strong priors over plausible crystal structures, but turning that prior into promising candidates requires steering toward several conflicting properties at once. Fine-tuning is expensive and produces a new model for every new objective, while differentiable guidance is not applicable to many real-world tasks. We instead adapt these generative models at inference time using Diffusion Tree Sampling (DTS)",
            "url": "https://arxiv.org/abs/2611.17462",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-ac203f62",
            "type": "article",
            "title": "Availability-Aware Historical Feedback Learning for Electric Arc Furnace Temperature and Chemical Composition Prediction",
            "summary": "arXiv:2611.17463v1 Announce Type: new Abstract: Electric arc furnace (EAF) steelmaking plays an increasingly important role in recycled and lower-carbon steel production. However, variability in charged materials, energy input, and operating conditions makes accurate prediction of molten steel temperature and final chemical composition challenging. Intermittent or delayed measurements limit the information available for timely prediction. This study develops an availability-aware historical-feed",
            "url": "https://arxiv.org/abs/2611.17463",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-231ae5e2",
            "type": "article",
            "title": "Improved Regret via Variance-Aware Policy Optimization in Adversarial MDPs",
            "summary": "arXiv:2611.17469v1 Announce Type: new Abstract: We study episodic adversarial MDPs with unknown dynamics and full-information rewards and achieve regret with leading term $\\widetilde O(\\sqrt{H^3SAK})$. Here $H$ is the horizon, $S$ the total number of states across decision layers, $A$ the number of actions, and $K$ the number of episodes. This improves the two incomparable prior leading terms by factors $\\sqrt{S/H}$ and $\\sqrt{H^3}$, leaving a factor $\\sqrt H$ to the lower-bound scale. We give a",
            "url": "https://arxiv.org/abs/2611.17469",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-935d8a28",
            "type": "article",
            "title": "Do Mixture-of-Experts Routers Benefit from Ensembling?",
            "summary": "arXiv:2611.17472v1 Announce Type: new Abstract: Ensembles of neural networks have repeatedly been shown to outperform single networks. Ensemble theory attributes this gain to the accuracy of the individual members and the diversity among them. Mixture-of-experts models already combine many experts, but a single router decides how the experts are weighted for each input. Motivated by ensemble theory, we apply ensembling to the routing step: an ensemble of routers, trained jointly with the experts",
            "url": "https://arxiv.org/abs/2611.17472",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-5108bce7",
            "type": "article",
            "title": "ERGo: Energy-Efficient Hybrid Graph Neural Network Training on Processing-In-Memory Architectures",
            "summary": "arXiv:2611.17478v1 Announce Type: new Abstract: Processing-in-memory (PIM) has been proposed as an alternative computing paradigm for training Deep Neural Networks, including Graph Neural Networks (GNNs). Despite these advancements, training GNN workloads on PIM devices necessitates off-chip memory access. This off-chip access is expensive in terms of performance and energy, thereby impacting the overall energy efficiency of the training process on PIM platforms. In this paper, we propose a nove",
            "url": "https://arxiv.org/abs/2611.17478",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-5406eadd",
            "type": "article",
            "title": "SURFACE: Preference Distribution Matching on Pareto Fronts via Scalarization Steering",
            "summary": "arXiv:2611.17484v1 Announce Type: new Abstract: Constructing a finite Pareto representation requires allocating a limited number of solutions across the Pareto front (PF). A desired preference distribution specifies this allocation by assigning more solutions to some PF regions and fewer to others. Scalarization generates Pareto solutions by varying objective weights and solving single-objective problems. However, a distribution of scalarization weights generally does not translate into the same",
            "url": "https://arxiv.org/abs/2611.17484",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-c10e426b",
            "type": "article",
            "title": "Identifying Latent Neural Dynamics with Recognition-Parametrised Gaussian Process Dynamical Systems",
            "summary": "arXiv:2611.17487v1 Announce Type: new Abstract: Neural circuits are densely interconnected systems, and many neural computations are thought to be mediated by the dynamical evolution of activity within these recurrent networks, a hypothesis often referred to as \"computation through dynamics\". Under this view, understanding computation within real neural populations depends on building models of the dynamical rules governing the temporal evolution of recorded high-dimensional neural activity. Mos",
            "url": "https://arxiv.org/abs/2611.17487",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-c62fa5f2",
            "type": "article",
            "title": "PF-Exec: Large-Scale UAV Relay Load Balancing with an Anchored Potential Direction Decider and a Per-Dimension Gain-Learning Executor",
            "summary": "arXiv:2611.17492v1 Announce Type: new Abstract: Relay load balancing in large-scale UAV networks requires joint decisions on association, bandwidth, power and hovering position, while the dimension grows with the number of UAVs and each UAV sees only its local state. This paper splits the decision into two layers. An anchored potential direction decider emits a closed-form target at every instant and returns its direction only. A gain-learning executor maps a local observation into a positive ga",
            "url": "https://arxiv.org/abs/2611.17492",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-b2969b32",
            "type": "article",
            "title": "Adversarially Batched Online Learning",
            "summary": "arXiv:2611.17500v1 Announce Type: new Abstract: We introduce adversarially batched online learning, where the learner may update its decision rule only at a restricted set of time steps. Equivalently, the time horizon is partitioned into batches, within which the learner must use a fixed policy. We study this model for four canonical problems: stochastic bandits, online convex optimization, prediction with expert advice, and adversarial bandits. For stochastic bandits, we identify a universal co",
            "url": "https://arxiv.org/abs/2611.17500",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-4c2c6384",
            "type": "article",
            "title": "A Synthetic Benchmark for Target-Centric Causal Discovery in Industrial Processes",
            "summary": "arXiv:2611.17556v1 Announce Type: new Abstract: Causal discovery promises to find which recorded variables drive a process target such as product quality, yield or energy use. Process data with known causes are rare, and in existing benchmarks the difficulty of identifying the causes of a target is either fixed by the recorded data or set for the whole graph. We present a synthetic benchmark for target-centric causal discovery in industrial processes. Its generator is a two-level structural caus",
            "url": "https://arxiv.org/abs/2611.17556",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-d9cee7a9",
            "type": "article",
            "title": "Generative Design with Real-time Physics Explainability: Towards Product Definition in the Age of AI",
            "summary": "arXiv:2611.17560v1 Announce Type: new Abstract: In this article, we explore opportunities from using generative AI in mechanical engineering. On the one hand, generative methods based on invertible neural networks have been recently proposed for the solution of inverse design problems, allowing to infer design parameters from performance labels. On the other hand, generative AI models have shown that they can act as real time capable surrogates for physics based simulations. Here we suggest that",
            "url": "https://arxiv.org/abs/2611.17560",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-38dc97f6",
            "type": "article",
            "title": "Beyond a Single Sign: BiTE for Branch-Preserving Binary Threshold Expansion",
            "summary": "arXiv:2611.17576v1 Announce Type: new Abstract: Binary neural networks (BNNs) reduce model storage and replace many arithmetic operations with efficient bitwise computations, but the single-sign activation limits their representational capacity. Each preactivation direction is compared with one threshold and reduced to one of two values, discarding variation within each sign region. We propose \\textbf{BiTE} (Branch-Preserving Binary Threshold Expansion), which evaluates each feature direction ag",
            "url": "https://arxiv.org/abs/2611.17576",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-46b3f642",
            "type": "article",
            "title": "Fair and Efficient Distributed EV Charging Coordination Using Threshold Cryptography and Machine Learning",
            "summary": "arXiv:2611.17577v1 Announce Type: new Abstract: Distributed coordination of electric vehicle (EV) charging with the alternating direction method of multipliers (ADMM) keeps EV constraints private but assumes that every participant follows the algorithm. This assumption fails in practice, where a biased operator or a man-in-the-middle (MITM) attacker can send inconsistent signals to different EVs, and compromised EVs can deviate from their prescribed updates. We first prove that attacks by such a",
            "url": "https://arxiv.org/abs/2611.17577",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-0020dd2a",
            "type": "article",
            "title": "LightTAR: Light and Fast Target-Aware Representations for Multimodal Tabular Learning",
            "summary": "arXiv:2611.17600v1 Announce Type: new Abstract: Multimodal tabular learning involves classification and regression on tables combining structured numerical and categorical features with unstructured text and images. These unstructured fields are commonly represented by pretrained encoder embeddings reduced with principal component analysis (PCA). This reduction prioritizes input variance but may overlook information relevant to predicting the target. We introduce LightTAR, a simple, inexpensive ",
            "url": "https://arxiv.org/abs/2611.17600",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-61241346",
            "type": "article",
            "title": "Offline KL-Regularized Reinforcement Learning with Stochastic Transitions",
            "summary": "arXiv:2611.17603v1 Announce Type: new Abstract: KL-regularized reinforcement learning is the objective behind RL fine-tuning of language models, and its data are often collected by the reference policy itself. However, beyond bandits and deterministic transitions, little is known about how much offline data it needs. In this work, we study finite discounted Markov decision processes with unknown stochastic transitions, $K$ state-action pairs and effective horizon $H$. For data that cover the reg",
            "url": "https://arxiv.org/abs/2611.17603",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-e2252d53",
            "type": "article",
            "title": "Neural Simulation-Based Inference for Computational Cognitive Models: A Tutorial",
            "summary": "arXiv:2611.17608v1 Announce Type: new Abstract: Computational cognitive modeling is an increasingly essential methodology to elucidate the mechanisms underlying human and animal cognition. By fitting models to empirical data, researchers can perform parameter estimation, model comparison, and model-based neural analyses to assess competing theories, quantify individual differences, and investigate neural representations. Typically, fitting these models relies on likelihood-dependent methods (e.g",
            "url": "https://arxiv.org/abs/2611.17608",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-cffc02a2",
            "type": "article",
            "title": "Tracing Knowledge and Reasoning in LLM",
            "summary": "arXiv:2611.17461v1 Announce Type: new Abstract: Current evaluations of Large Language Models (LLMs) report performance at the dataset or task level, revealing what a model achieves but offering little insight into the internal processing associated with its performance. We propose a task-centric framework for characterising the model-internal processing elicited by a benchmark. It combines attention- and activation-based signals with representation-based measures to characterise benchmarks spann",
            "url": "https://arxiv.org/abs/2611.17461",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-6bc53ef6",
            "type": "article",
            "title": "Poetry 'resurrection' from LLMs",
            "summary": "arXiv:2611.17464v1 Announce Type: new Abstract: In an era where linguistic creativity is challenged by algorithmic authorship, we pose a provocative question: can artificial intelligence (AI) write an infinite poem? We set up two experiments to investigate the quantitative and qualitative output produced by AI utilizing couplets that evidence a poetic relationship between end words, the only variation in a strict anaphoric, litany structure. Our results show that while AI can generate interestin",
            "url": "https://arxiv.org/abs/2611.17464",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-4fb37acb",
            "type": "article",
            "title": "Creating Steering Vectors via Input Optimisation",
            "summary": "arXiv:2611.17481v1 Announce Type: new Abstract: Activation steering influences a language model's behaviour by adding a steering vector to its residual stream during generation, but it requires white-box access to modify the hidden states at every generation step. We introduce ActGCG, a modification of Greedy Coordinate Gradient which optimises a suffix to match the target activations obtained by adding a steering vector to the residual stream. We further introduce universal suffixes trained acr",
            "url": "https://arxiv.org/abs/2611.17481",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-a9618a54",
            "type": "article",
            "title": "Benchmarks Don't Talk Back: Learning Through Self-Disagreement",
            "summary": "arXiv:2611.17563v1 Announce Type: new Abstract: A benchmark usually measures a language model alone with a question, and deployment rarely looks like that: answers get contested, and models give up correct ones under counter-reasoning that is itself wrong. We show that what fails there is not a shortfall of accuracy but a capability in its own right -- knowing when to hold a position and when to yield to a better one -- which we call dialectical competence. It stands on first-answer accuracy wit",
            "url": "https://arxiv.org/abs/2611.17563",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-9aa193d3",
            "type": "article",
            "title": "CamSpokenNLI: A Pilot Resource for Cross-Variety Transfer Evaluation in Cameroon Pidgin English and Camfranglais",
            "summary": "arXiv:2611.17573v1 Announce Type: new Abstract: Cameroon Pidgin English (a West African Pidgin English variety) and Camfranglais are widely used in Cameroon, yet to our knowledge, neither has a dedicated spoken Natural Language Inference (NLI) resource. We introduce CamSpokenNLI, a pilot dataset of 114 scenario-matched spoken NLI items in each variety (228 in total), recorded by one native speaker from Buea. Each item pairs premise audio and text with a hypothesis, an NLI label, and English glos",
            "url": "https://arxiv.org/abs/2611.17573",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-1e6da13c",
            "type": "article",
            "title": "Small Open-Weight Language Models for Crowd Decision-Making: Capabilities and Limits of Output-Level Fusion and Linguistic Aggregation",
            "summary": "arXiv:2611.17589v1 Announce Type: new Abstract: Small open-weight language models can run crowd decision-making (CDM) on an institution's own infrastructure, which keeps the checkpoint, the inference settings and the opinions under its control. Their judgements vary across repeated queries; whether repeating and aggregating them, often through the 2-tuple linguistic representation, settles the decision rather than the labels is untested. Five open-weight models of 8-36B total parameters on one c",
            "url": "https://arxiv.org/abs/2611.17589",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-70fb34e7",
            "type": "article",
            "title": "What Do Multilingual Self-Supervised Speech Encoders Know About Sentence Meaning?",
            "summary": "arXiv:2611.17686v1 Announce Type: new Abstract: Self-supervised speech models (S3Ms) have been scaled up to thousands of languages and millions of hours of audio, yet little is known about how these models structure semantic knowledge across languages. In this work, we probe sentence-level semantics in massively multilingual S3Ms across 76 languages, using linear maps from frozen speech representations into a frozen text embedding space, evaluated by cross-modal retrieval. Sentence meaning is li",
            "url": "https://arxiv.org/abs/2611.17686",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-32151157",
            "type": "article",
            "title": "See Every Block Once: Variance-Reduced Post-Training for Masked Diffusion Language Models",
            "summary": "arXiv:2611.17754v1 Announce Type: new Abstract: Policy gradient methods are widely used to align large language model outputs with human preferences, but the log-likelihoods they require cannot be directly computed in Masked Diffusion Language Models (MDLMs). MDLM post-training therefore relies on a surrogate objective based on the Evidence Lower Bound (ELBO), and recent work estimates a blockwise variant of this objective via Monte Carlo methods to support blockwise generation. However, existin",
            "url": "https://arxiv.org/abs/2611.17754",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-a81b9f8f",
            "type": "article",
            "title": "GDP.xlsx: Benchmarking Frontier Models on Professional Workbooks",
            "summary": "arXiv:2611.17766v1 Announce Type: new Abstract: We introduce GDP_xlsx, a benchmark of 70 realistic workplace requests written by professionals in 13 domains, each paired with an expert rubric of binary criteria and with the complete workbook or CSV file it refers to. Together the files hold 2.18 million populated cells, and the rubrics 791 criteria. Agents work in a container with spreadsheet tools, and an agentic judge that can open the file applies the rubric. Completeness and exclusivity crit",
            "url": "https://arxiv.org/abs/2611.17766",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-15fdb974",
            "type": "article",
            "title": "Activation Steering the Nan-Shan in Typhoon",
            "summary": "arXiv:2611.17768v1 Announce Type: new Abstract: Typhoon is an audiovisual installation based on Joseph Conrad's novella, in which Captain MacWhirr steers the steamer Nan-Shan directly into and through a catastrophic storm. The novella is a study of what happens when the scale of a phenomenon exceeds the practical and conceptual frameworks available to comprehend it and of the paralysis that sometimes follows a demand for action in such overwhelming circumstances. We chose the text of Typhoon to ",
            "url": "https://arxiv.org/abs/2611.17768",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-56ecbf67",
            "type": "article",
            "title": "LuthorSQL: A One-Pass Text-to-SQL Model as System 1 for a Slower LLM Agent",
            "summary": "arXiv:2611.17780v1 Announce Type: new Abstract: Frontier LLM agents that query databases with tools are among the most accurate text-to-SQL systems, but per-query calls to a hosted model are slow, costly, and expose sensitive organizational queries to external APIs. For example, a Claude Opus 5.5 agent serving 10,000 queries a day costs over $70,000 a year at list price. To what extent can local hardware already owned by an organization offload this burden without compromising the agent's accura",
            "url": "https://arxiv.org/abs/2611.17780",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-9ccb449f",
            "type": "article",
            "title": "RAG-FGO: Dynamic Optimization of Query-Rewriting Agents for Generative RAG",
            "summary": "arXiv:2611.17788v1 Announce Type: new Abstract: In retrieval-augmented generation (RAG), query rewriting serves as a core interface that aligns user intent with external knowledge bases, and its effectiveness directly determines overall system performance. However, generative retrieval faces three critical challenges in real-world deployment: noisy retrieval feedback, the difficulty of jointly optimizing discrete prompt templates and continuous decoding-control parameters, and the limited genera",
            "url": "https://arxiv.org/abs/2611.17788",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-d6c99d69",
            "type": "article",
            "title": "CarryOn: Learning What to Keep in Context for Long-Horizon LLM Agents",
            "summary": "arXiv:2611.17808v1 Announce Type: new Abstract: Long-horizon LLM agents can coordinate tools and applications to complete complex, multi-step tasks. As their interaction histories grow, effective context management becomes essential for reliable and efficient execution. Existing methods reduce context through summarization, pruning, or history selection. However, shorter contexts do not necessarily support better decisions: summaries can omit exact details needed for later actions, while retaini",
            "url": "https://arxiv.org/abs/2611.17808",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-abcf06f5",
            "type": "article",
            "title": "AI Reviews Are Biased Towards the Appendix",
            "summary": "arXiv:2611.17811v1 Announce Type: new Abstract: Large language models (LLMs) are increasingly used by reviewers to highlight weaknesses during the reviewing process, and their outputs often shape the reviews that humans submit. Prior work studies how LLM reviews align with human ones, yet we know little about which parts of a paper these models actually criticize. In this work, we investigate 2,000 papers from ACL, EMNLP, COLM, and ICLR, and show that the latest GPT and Claude models draw about ",
            "url": "https://arxiv.org/abs/2611.17811",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-0ac5c054",
            "type": "article",
            "title": "PaperCritic: Training AI Reviewers to Assess the Soundness of Papers",
            "summary": "arXiv:2611.17834v1 Announce Type: new Abstract: As AI conference submissions grow rapidly, AI reviewers are being considered as an alternative. However, simply imitating human reviewing is problematic because reviewing serves multiple purposes, including evaluating papers' soundness, interestingness, and clarity, the last two of which are subjective. In this work, we focus solely on evaluating soundness, the most important and objective criterion, and define the learning objective as jointly max",
            "url": "https://arxiv.org/abs/2611.17834",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-b4fbfe19",
            "type": "article",
            "title": "BLiMP-RO: A Benchmark of Linguistic Minimal Pairs for Romanian",
            "summary": "arXiv:2611.17836v1 Announce Type: new Abstract: Evaluating grammatical knowledge in language models requires benchmarks that cover diverse linguistic phenomena and are validated against human judgments. We introduce BLiMP-RO, a Romanian minimal-pair benchmark covering 50 grammatical phenomena across ten domains. It comprises 493 linguist-constructed sentence pairs, evaluated by native speakers in five experiments, and 5,942 additional semi-automatically generated, human-validated pairs. We furth",
            "url": "https://arxiv.org/abs/2611.17836",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-ba492d0f",
            "type": "article",
            "title": "UltraBERT-JEV: Encoder models for System One Applications",
            "summary": "arXiv:2611.17867v1 Announce Type: new Abstract: We study encoder models that pick one of several options in a single forward pass, without generating text. This technical report compares recipes for fine-tuning UltraBERT-380M with one mask per option. The four recipes differ in how an option is scored and how the scores become probabilities; one follows the expert gate of DeepSeek-V3. Trained on the same data as Qwen3-0.6B with the LLM-as-Jev recipe (67.6% test accuracy), the encoder reaches 69.",
            "url": "https://arxiv.org/abs/2611.17867",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-cf88cd4a",
            "type": "article",
            "title": "MineralExp: Benchmarking LLM Agents for Experimental Design in Real-World Mineral Exploration",
            "summary": "arXiv:2611.17931v1 Announce Type: new Abstract: In mineral exploration, verifying geological hypotheses relies on designing a variety of geochemical experiments. While large language model (LLM) agents increasingly automate downstream geoscientific data analysis, a crucial upstream capability remains largely unexplored: automated geochemical experimental design, i.e., deciding which experiments to conduct before committing to costly laboratory work. Evaluating this capability of LLM agents is di",
            "url": "https://arxiv.org/abs/2611.17931",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-ee6a189c",
            "type": "article",
            "title": "Can You Trust Your Finance Agent? Drift, Diagnosis and Auditable Intervention on One Shipped Financial Research Agent",
            "summary": "arXiv:2611.17939v1 Announce Type: new Abstract: Can you trust a finance agent you did not build? We take the reference agent shipped with the Vals Finance Agent Benchmark v2, leave it untuned, and ask four questions a leaderboard score does not answer. Is the score stable? Seven identical runs on 18 questions pass between 105 and 123 of 149 rubric criteria, and the bill varies by up to 17%. When the agent fails, who owns the fix? We classify 20 failed trajectories, from three runs of nine held-o",
            "url": "https://arxiv.org/abs/2611.17939",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-da69039a",
            "type": "article",
            "title": "In the Eye of the Beholder: Human Label Variation in Textual and Visual Entailment",
            "summary": "arXiv:2611.17954v1 Announce Type: new Abstract: Human Label Variation (HLV) is increasingly recognised as an informative signal that reflects meaningful differences in how people interpret the same input. While HLV has been studied extensively in Natural Language Inference (NLI), much less is known about HLV in Visual Entailment (VE; Xie et al., 2019), where the textual premise is replaced by an image, keeping the hypothesis and label space fixed. We introduce VariNLI-VE, a dataset for studying ",
            "url": "https://arxiv.org/abs/2611.17954",
            "source": "arxiv",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-3e4a4b87",
            "type": "article",
            "title": "New Ask Maps features to help with holiday travel and prep",
            "summary": "tbd",
            "url": "https://blog.google/products-and-platforms/products/maps/ask-maps-holiday-travel-features-2026/",
            "source": "blogs",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-3987a872",
            "type": "article",
            "title": "Google Search gets a major upgrade with Gemini 4 Hydrogen",
            "summary": "Rotating carousel of cards with text \"Gemini 4 Hydrogen\" \"Now in Google Search\" and examples from the Google Search UI",
            "url": "https://blog.google/products-and-platforms/products/search/gemini-4-hydrogen-search-upgrade/",
            "source": "blogs",
            "date": "2026-11-12",
            "trendingScore": 50
        },
        {
            "id": "article-3586fd5c",
            "type": "article",
            "title": "Gemini Decisions API: Fast, calibrated multimodal decisions",
            "summary": "Meet the Gemini Decisions API, powered by our state-of-the-art model Gemini Decision for fast, calibrated multimodal decisions.",
            "url": "https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/",
            "source": "blogs",
            "date": "2026-11-11",
            "trendingScore": 50
        },
        {
            "id": "article-d3a6593e",
            "type": "article",
            "title": "Try Tabletop: Customizable, collaborative spaces for groups to plan, play and do more together.",
            "summary": "Tabletop is a new Google Labs experiment for group collaboration. Build interactive itineraries, polls, and brackets with your team. Join the waitlist.",
            "url": "https://blog.google/innovation-and-ai/models-and-research/google-labs/tabletop/",
            "source": "blogs",
            "date": "2026-11-11",
            "trendingScore": 50
        },
        {
            "id": "article-651a0fcc",
            "type": "article",
            "title": "Experience the art and legacy of Dr. Esther Mahlangu",
            "summary": "A portrait of Dr Nontobeko Ntombela, Senior Lecturer, Department of Curatorial, Public and Visual Cultures, Wits School of Arts, University of the Witwatersrand",
            "url": "https://blog.google/company-news/outreach-and-initiatives/arts-culture/esther-mahlangu/",
            "source": "blogs",
            "date": "2026-11-11",
            "trendingScore": 50
        },
        {
            "id": "article-8477a815",
            "type": "article",
            "title": "Gemini Intelligence: New features, on more devices",
            "summary": "A man speaks into a phone. Text on the screen reads 'Gemini Intelligence'",
            "url": "https://blog.google/products-and-platforms/platforms/android/gemini-intelligence-november-features-devices/",
            "source": "blogs",
            "date": "2026-11-10",
            "trendingScore": 50
        },
        {
            "id": "article-0b514ad9",
            "type": "article",
            "title": "We\u2019re bringing Adobe\u2019s creative and productivity tools closer to your everyday Google apps.",
            "summary": "Google and Adobe are integrating creative and productivity tools. Edit PDFs in Drive and use Adobe AI in Gemini. See what\u2019s new in this partnership.",
            "url": "https://blog.google/company-news/inside-google/company-announcements/adobe-google-partnership-integration/",
            "source": "blogs",
            "date": "2026-11-10",
            "trendingScore": 50
        },
        {
            "id": "article-da8db646",
            "type": "article",
            "title": "Google Cloud launches Gemini for healthcare.",
            "summary": "Google Cloud launches Gemini for healthcare to unify clinical data and automate workflows. Reduce administrative burnout. See how it works.",
            "url": "https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/gemini-for-healthcare/",
            "source": "blogs",
            "date": "2026-11-10",
            "trendingScore": 50
        },
        {
            "id": "article-9c477938",
            "type": "article",
            "title": "Build a Robot Pick-and-Place Simulation with SimReady Assets",
            "summary": "",
            "url": "https://huggingface.co/blog/nvidia/simready-assets-robot-pick-and-place",
            "source": "blogs",
            "date": "2026-11-13",
            "trendingScore": 50
        },
        {
            "id": "article-e9d93e60",
            "type": "article",
            "title": "XYBench: Helping with the problem behind the question",
            "summary": "",
            "url": "https://huggingface.co/blog/allenai/xybench",
            "source": "blogs",
            "date": "2026-11-13",
            "trendingScore": 50
        },
        {
            "id": "article-1f76189b",
            "type": "article",
            "title": "llama.cpp v1.0.0 : A stronger foundation for the local AI ecosystem",
            "summary": "",
            "url": "https://huggingface.co/blog/ggml-org/llama-cpp-v1",
            "source": "blogs",
            "date": "2026-11-13",
            "trendingScore": 50
        },
        {
            "id": "article-f052c684",
            "type": "article",
            "title": "LLMON: An LLM-Native Markup Language, now in Granite 4.1 3B and 8B",
            "summary": "",
            "url": "https://huggingface.co/blog/ibm-granite/llmon-an-llm-native-markup-language-now-in-granite",
            "source": "blogs",
            "date": "2026-11-12",
            "trendingScore": 50
        },
        {
            "id": "article-d3916f44",
            "type": "article",
            "title": "The Definitive Fine-Tuning Guide for 3D Generation Models",
            "summary": "",
            "url": "https://huggingface.co/blog/3d-lora-finetuning",
            "source": "blogs",
            "date": "2026-11-12",
            "trendingScore": 50
        },
        {
            "id": "article-4c5ef8a2",
            "type": "article",
            "title": "BotPsy: On-Device Personal Assistant with a Sub-Billion-Parameter Agentic Language Model",
            "summary": "",
            "url": "https://huggingface.co/blog/qvac/botpsy",
            "source": "blogs",
            "date": "2026-11-11",
            "trendingScore": 50
        },
        {
            "id": "article-542cd5af",
            "type": "article",
            "title": "Token Pooling-Aware Training for Multi-Vector Embedding Models",
            "summary": "",
            "url": "https://huggingface.co/blog/token-pooling",
            "source": "blogs",
            "date": "2026-11-11",
            "trendingScore": 50
        },
        {
            "id": "article-992e8c41",
            "type": "article",
            "title": "Pay for Hugging Face Hub with your Google Cloud Account",
            "summary": "",
            "url": "https://huggingface.co/blog/hub-gcp-marketplace",
            "source": "blogs",
            "date": "2026-11-10",
            "trendingScore": 50
        },
        {
            "id": "article-de10773c",
            "type": "article",
            "title": "Decision Models in Hugging Face",
            "summary": "",
            "url": "https://huggingface.co/blog/decision",
            "source": "blogs",
            "date": "2026-11-10",
            "trendingScore": 50
        },
        {
            "id": "article-0f553794",
            "type": "article",
            "title": "Building AI that can work across the enterprise",
            "summary": "For years, enterprises have approached AI as a series of discrete experiments. Now, as AI agents begin to move into the workflows that run the business, that model is starting to look inadequate. Organizations moving from isolated pilots toward AI agents and autonomous workflows are confronting a deeper challenge: rethinking how data, business processes, governance,&#8230;",
            "url": "https://www.technologyreview.com/2026/11/16/1146070/building-ai-that-can-work-across-the-enterprise/",
            "source": "blogs",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-00fbeb28",
            "type": "article",
            "title": "Nvidia privately confirmed it could track chips to fight smuggling, even as industry publicly questioned feasibility",
            "summary": "Representatives from Nvidia told Illinois representative Bill Foster that it would be technically possible to implement the anti-smuggling measures called for in the Chip Security Act, Foster told MIT Technology Review in an interview on Thursday, even while an industry group claimed they were \u201cpotentially infeasible.\u201d Foster, a Democrat who serves on the Committee on&#8230;",
            "url": "https://www.technologyreview.com/2026/11/12/1148483/nvidia-chip-tracking/",
            "source": "blogs",
            "date": "2026-11-12",
            "trendingScore": 50
        },
        {
            "id": "article-1127808f",
            "type": "article",
            "title": "The backlash about data centers might really just be about electricity",
            "summary": "This story originally appeared in The Algorithm, our weekly newsletter on AI. To get stories like this in your inbox first, sign up here. As voters went to the polls across the US last week, I was in Ashburn, Virginia, part of what\u2019s known as Data Center Alley\u2014the world\u2019s largest data center hub. It\u2019s also where&#8230;",
            "url": "https://www.technologyreview.com/2026/11/09/1148194/the-backlash-about-data-centers-might-really-just-be-about-electricity/",
            "source": "blogs",
            "date": "2026-11-09",
            "trendingScore": 50
        },
        {
            "id": "article-9541439d",
            "type": "article",
            "title": "Nancy Pelosi\u2019s replacement is ready to regulate AI",
            "summary": "As a California state senator, Scott Wiener authored legislation aimed at curbing the risks of advanced AI. Now, after winning a tight congressional race for former Speaker of the House Nancy Pelosi\u2019s open San Francisco seat, he could play a key role in setting federal AI policy.&#160; I spoke with Wiener on Friday, only a&#8230;",
            "url": "https://www.technologyreview.com/2026/11/09/1148182/nancy-pelosis-replacement-is-ready-to-regulate-ai/",
            "source": "blogs",
            "date": "2026-11-09",
            "trendingScore": 50
        },
        {
            "id": "article-42d6602d",
            "type": "article",
            "title": "Caption.me AI Arena \u2013 How Funny Are LLMs?",
            "summary": "",
            "url": "https://www.caption.me/ai-arena",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-efab56c7",
            "type": "article",
            "title": "Cartoons on an AI PMs' shifting choices in a low-attention post-AGI era",
            "summary": "",
            "url": "https://boskykode.com/blog/low-attention-post-agi-world-contrarian-product-perspectives",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-83f52ff8",
            "type": "article",
            "title": "Where Are Our 10\u00d7 Engineers? The Math of AI-Assisted Coding [video]",
            "summary": "",
            "url": "https://www.youtube.com/watch?v=go_Fjw_51W4",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-38c0b09f",
            "type": "article",
            "title": "Plzdontkillus \u2013 experimental creator bootcamp about AI doom",
            "summary": "",
            "url": "https://plzdontkillus.com/",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-bad9b100",
            "type": "article",
            "title": "AI Video to Text \u2013 Free Online Video to Text and Audio to Text Converter",
            "summary": "",
            "url": "https://www.aivideototext.org/",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-24cc6e0b",
            "type": "article",
            "title": "Build or Buy Manufacturing ERP in the Age of AI",
            "summary": "",
            "url": "https://www.paxerp.com/blog/build-or-buy-manufacturing-erp-ai",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-dcc83dc9",
            "type": "article",
            "title": "A human-steered, AI-built freeware hybrid of Ableton, Bitwig, and Renoise",
            "summary": "",
            "url": "https://edu.morningdewmedia.com/product/morningdaw",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-acc2f8d5",
            "type": "article",
            "title": "Guilty Until Proven Human: How AI Suspicion Broke Book Publishing",
            "summary": "",
            "url": "https://smarterarticles.co.uk/guilty-until-proven-human-how-ai-suspicion-broke-book-publishing",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-9dbd5792",
            "type": "article",
            "title": "Nationwide ground stop exposes the nightmare of flying from US airports",
            "summary": "",
            "url": "https://english.elpais.com/usa/2026-11-16/nationwide-ground-stop-exposes-the-nightmare-of-flying-from-us-airports.html",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-756341d4",
            "type": "article",
            "title": "Nvidia ordered to explain $12B supply of AI chips to China",
            "summary": "",
            "url": "https://www.thetimes.com/business/companies-markets/article/nvidia-jensen-huang-us-treasury-ai-chips-china-ckwd9037f",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-dfff9e74",
            "type": "article",
            "title": "Google: AI agent turned a security patch into a exploit with no human help",
            "summary": "",
            "url": "https://www.theregister.com/research/2026/11/16/google-ai-agent-turned-a-security-patch-into-a-working-exploit-with-no-human-help/5309926",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-0c8877bb",
            "type": "article",
            "title": "The Icarus Files: AI labs are aggressively diversifying away from Nvidia",
            "summary": "",
            "url": "https://am.jpmorgan.com/us/en/asset-management/institutional/insights/market-insights/eye-on-the-market/the-icarus-files-whats-next-for-nvidia-and-ai/",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-3fcb5c99",
            "type": "article",
            "title": "Challenges for human oversight in detecting AI-assisted diagnostic malfunctions",
            "summary": "",
            "url": "https://www.nature.com/articles/s41746-026-03410-x",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-972933ed",
            "type": "article",
            "title": "AI-driven scientific progress threatened by an obsession",
            "summary": "",
            "url": "https://www.unite.ai/ai-driven-scientific-progress-threatened-by-a-strange-obsession/",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-9a4798b6",
            "type": "article",
            "title": "Reverse-Engineering Kevin J. Lang's Lost OCaml Code in Apache DataSketches",
            "summary": "",
            "url": "https://www.tisonkun.io/posts/ai-repair-program/",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-ef74113b",
            "type": "article",
            "title": "US Moves to Limit Remote Access to Advanced AI Chips",
            "summary": "",
            "url": "https://www.bloomberg.com/news/articles/2026-11-16/us-moves-to-limit-remote-access-to-advanced-ai-chips",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-a1df07d5",
            "type": "article",
            "title": "Retain control over the writing process with LLMs [video]",
            "summary": "",
            "url": "https://www.youtube.com/shorts/GrgpK1KhsJ8",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-5f4b98ca",
            "type": "article",
            "title": "Irrigator: Local AI agent orchestration with recovery and spending limits",
            "summary": "",
            "url": "https://irrigator.dev/",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-e20ebde7",
            "type": "article",
            "title": "Show HN: OpenButterfly \u2013 Tracking the environmental impact of the AI ecosystem",
            "summary": "",
            "url": "https://openbutterfly.com",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-f42b77ca",
            "type": "article",
            "title": "Show HN: Storm - self-hosted assistant that builds apps and waits for your OK",
            "summary": "",
            "url": "https://github.com/realentity/realentity-storm",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-58482a10",
            "type": "article",
            "title": "Show HN: Neuronte \u2013 small neural nets as a font",
            "summary": "",
            "url": "https://github.com/itsmygithubacct/neuronte-v2",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-a3e3f29b",
            "type": "article",
            "title": "I got Claude to rewrite curl in Rust in ~4 weeks",
            "summary": "",
            "url": "https://claude.ai/artifact/K4BoWt35HQz1xWZBR3bMRD",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-0ec0016f",
            "type": "article",
            "title": "Theoretical Limits of Language Model Alignment",
            "summary": "",
            "url": "https://machinelearning.apple.com/research/theoretical-limits-alignment",
            "source": "hackernews",
            "date": "2026-11-13",
            "trendingScore": 50
        },
        {
            "id": "article-e1a45ad6",
            "type": "article",
            "title": "The Complete Guide to LLM Evaluation",
            "summary": "",
            "url": "https://machinelearningmastery.com/the-complete-guide-to-llm-evaluation/",
            "source": "hackernews",
            "date": "2026-11-13",
            "trendingScore": 50
        },
        {
            "id": "article-1ac99517",
            "type": "article",
            "title": "Gemini for Healthcare",
            "summary": "",
            "url": "https://cloud.google.com/blog/products/ai-machine-learning/gemini-for-healthcare",
            "source": "hackernews",
            "date": "2026-11-13",
            "trendingScore": 50
        },
        {
            "id": "article-4a0b225d",
            "type": "article",
            "title": "Show HN: PongArena \u2013 Play Pong and see how tiny neural networks think",
            "summary": "",
            "url": "https://jonaslindblom.github.io/pongarena/",
            "source": "hackernews",
            "date": "2026-11-15",
            "trendingScore": 50
        },
        {
            "id": "article-7fc512e4",
            "type": "article",
            "title": "Visualizing connections in neural networks across various modalities",
            "summary": "",
            "url": "https://nathanclonts.com/visualizing-neural-networks/",
            "source": "hackernews",
            "date": "2026-11-15",
            "trendingScore": 50
        },
        {
            "id": "article-8cd2858e",
            "type": "article",
            "title": "Anthropic,OpenAI sued for allegedly infringing inventors' neural network patents",
            "summary": "",
            "url": "https://www.reuters.com/legal/litigation/anthropic-sued-allegedly-infringing-inventors-neural-network-patents-2026-11-12/",
            "source": "hackernews",
            "date": "2026-11-13",
            "trendingScore": 50
        },
        {
            "id": "article-103afb4a",
            "type": "article",
            "title": "Hopfield and Bam \u2013 Neural Networks You Can Train and Use on Pen and Paper",
            "summary": "",
            "url": "https://blog.demofox.org/2026/11/12/hopfield-bam-neural-networks-you-can-train-and-use-on-pen-and-paper-with-integers/",
            "source": "hackernews",
            "date": "2026-11-13",
            "trendingScore": 50
        },
        {
            "id": "article-f19ea4dc",
            "type": "article",
            "title": "A new neural network input module improves humanoid locomotion",
            "summary": "",
            "url": "https://news.engin.umich.edu/2026/11/prism-a-neural-network-component-for-better-robotic-motor-control/",
            "source": "hackernews",
            "date": "2026-11-13",
            "trendingScore": 50
        },
        {
            "id": "article-af50ee33",
            "type": "article",
            "title": "AI Models Can Tell When Researchers Plant Concepts in Their Neural Networks",
            "summary": "",
            "url": "https://singularityhub.com/2026/11/11/ai-models-can-tell-when-researchers-plant-concepts-in-their-neural-networks-is-it-self-reflection/",
            "source": "hackernews",
            "date": "2026-11-11",
            "trendingScore": 50
        },
        {
            "id": "article-efc0a0db",
            "type": "article",
            "title": "Nvidia Kumo brings graph neural networks to Snowflake ML",
            "summary": "",
            "url": "https://medium.com/@yanivleven/the-gpu-meter-starts-when-you-forget-it-be078a94d7eb",
            "source": "hackernews",
            "date": "2026-11-11",
            "trendingScore": 50
        },
        {
            "id": "article-6d9bd315",
            "type": "article",
            "title": "Show HN: Usul \u2013 Express neural networks as raw math in C++23",
            "summary": "",
            "url": "https://github.com/crushr3sist/Usul-cpp",
            "source": "hackernews",
            "date": "2026-11-10",
            "trendingScore": 50
        },
        {
            "id": "article-fc3aa97d",
            "type": "article",
            "title": "Boris Cherny, the engineer behind Claude Code",
            "summary": "",
            "url": "https://www.cnn.com/2026/11/15/tech/anthropic-claude-code-boris-cherny",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-7aee8ee7",
            "type": "article",
            "title": "EasyPlanning \u2013 run a tour company's guide schedule from Claude",
            "summary": "",
            "url": "https://easyplanning.io",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-cc2d4e61",
            "type": "article",
            "title": "Show HN: Fifty Shades of Claude (Book as a Skill)",
            "summary": "",
            "url": "https://github.com/adi-devv/fifty-shades-of-claude",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-f009549c",
            "type": "article",
            "title": "How to clean up the worktrees Claude Code, Cursor and Codex leave behind",
            "summary": "",
            "url": "https://mopup.online/guides/coding-agent-worktrees",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-90b9361a",
            "type": "article",
            "title": "Show HN: Brief2mp4 \u2013 ClaudeCode/Codex skill that checks videos against the brief",
            "summary": "",
            "url": "https://github.com/qualixar/brief2mp4",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-7918eeb1",
            "type": "article",
            "title": "Show HN: Agent-Otel-Kit \u2013 OTel Dashboards for Claude Code, Codex and Gemini CLI",
            "summary": "",
            "url": "https://github.com/s3onghyun/agent-otel-kit",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-6270073a",
            "type": "article",
            "title": "Show HN: Redacted - Stop Handing Claude Your Secrets",
            "summary": "",
            "url": "https://github.com/svn-arv/redacted/",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-60661990",
            "type": "article",
            "title": "CloakPrompt \u2013 Local browser firewall that scrubs PII/keys before ChatGPT/Claude",
            "summary": "",
            "url": "https://addons.mozilla.org/en-US/firefox/addon/cloakprompt/",
            "source": "hackernews",
            "date": "2026-11-15",
            "trendingScore": 50
        },
        {
            "id": "article-d9e633e6",
            "type": "article",
            "title": "Ask HN: What do you include in your Claude/Codex instructions?",
            "summary": "",
            "url": "https://news.ycombinator.com/item?id=50509044",
            "source": "hackernews",
            "date": "2026-11-15",
            "trendingScore": 50
        },
        {
            "id": "article-90c609f8",
            "type": "article",
            "title": "Show HN: Claudette - Mistral models in Claude Code",
            "summary": "",
            "url": "https://github.com/raed667/claudette",
            "source": "hackernews",
            "date": "2026-11-15",
            "trendingScore": 50
        },
        {
            "id": "article-c3410c10",
            "type": "article",
            "title": "Ask HN: How does your AI assistant have a personality?",
            "summary": "",
            "url": "https://news.ycombinator.com/item?id=50508220",
            "source": "hackernews",
            "date": "2026-11-15",
            "trendingScore": 50
        },
        {
            "id": "article-7eef33e3",
            "type": "article",
            "title": "Show HN: Simple tool for FDEs to avoid missing HITL steps in claude code",
            "summary": "",
            "url": "https://github.com/robertodf99/hook-me-up",
            "source": "hackernews",
            "date": "2026-11-15",
            "trendingScore": 50
        },
        {
            "id": "article-86468cda",
            "type": "article",
            "title": "What music does Gemini Decision 1 like?",
            "summary": "",
            "url": "https://blog.lwarfield.dev/what-music-does-decision-gemini-like/",
            "source": "hackernews",
            "date": "2026-11-16",
            "trendingScore": 50
        },
        {
            "id": "article-76f26272",
            "type": "article",
            "title": "Catch Go Bugs That Fuzzing Misses with Gemini and a Constraint Solver",
            "summary": "",
            "url": "https://developers.googleblog.com/en/catch-go-bugs-that-fuzzing-misses-with-gemini-and-a-constraint-solver/",
            "source": "hackernews",
            "date": "2026-11-15",
            "trendingScore": 50
        },
        {
            "id": "article-2eb502da",
            "type": "article",
            "title": "Show HN:LetThemBuild:Claude, Codex, Gemini, Grok & OpenRouter models in Parallel",
            "summary": "",
            "url": "https://letthembuild.com",
            "source": "hackernews",
            "date": "2026-11-14",
            "trendingScore": 50
        },
        {
            "id": "article-7a5bff98",
            "type": "article",
            "title": "Ask HN: Is there a way to use Gemini API Keys with Antigravity?",
            "summary": "",
            "url": "https://news.ycombinator.com/item?id=50499317",
            "source": "hackernews",
            "date": "2026-11-14",
            "trendingScore": 50
        },
        {
            "id": "article-aea374a9",
            "type": "article",
            "title": "Get Started with Gemini 4: Explicit Context Caching Not Supported",
            "summary": "",
            "url": "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/get-started-with-gemini-4",
            "source": "hackernews",
            "date": "2026-11-14",
            "trendingScore": 50
        },
        {
            "id": "article-2925fc76",
            "type": "article",
            "title": "The anti-marketing aspect of the Gemini network",
            "summary": "",
            "url": "https://ploum.net/2026-11-12-anti-marketing-gemini.html",
            "source": "hackernews",
            "date": "2026-11-13",
            "trendingScore": 50
        },
        {
            "id": "article-8ae64a0d",
            "type": "article",
            "title": "Gemini 4 Hydrogen (High) Intelligence, Performance and Price Analysis",
            "summary": "",
            "url": "https://artificialanalysis.ai/models/gemini-4-hydrogen-high",
            "source": "hackernews",
            "date": "2026-11-13",
            "trendingScore": 50
        },
        {
            "id": "article-9ea4c258",
            "type": "article",
            "title": "Intelligent Eyewear with Gemini",
            "summary": "",
            "url": "https://glasses.google/",
            "source": "hackernews",
            "date": "2026-11-13",
            "trendingScore": 50
        },
        {
            "id": "article-b3f93f30",
            "type": "article",
            "title": "Comparing Jev, Gemini, and OpenAI decision models",
            "summary": "",
            "url": "https://blog.slopcop.com/comparing-jev-gemini-and-openai-decision-models/",
            "source": "hackernews",
            "date": "2026-11-12",
            "trendingScore": 50
        },
        {
            "id": "article-250664fb",
            "type": "article",
            "title": "Gemini 4 Hydrogen and Orchestrator",
            "summary": "",
            "url": "https://antigravity.google/blog/gemini-4-hydrogen-in-google-antigravity/",
            "source": "hackernews",
            "date": "2026-11-12",
            "trendingScore": 100
        },
        {
            "id": "article-9afe5c76",
            "type": "article",
            "title": "Gemini 4 Hydrogen (Xhigh) Intelligence, Performance and Price Analysis",
            "summary": "",
            "url": "https://artificialanalysis.ai/models/gemini-4-hydrogen",
            "source": "hackernews",
            "date": "2026-11-12",
            "trendingScore": 50
        },
        {
            "id": "article-a6377243",
            "type": "article",
            "title": "Gemini 4 Hydrogen",
            "summary": "",
            "url": "https://aistudio.google.com/docs/whats-new-gemini-4-hydrogen",
            "source": "hackernews",
            "date": "2026-11-12",
            "trendingScore": 52
        },
        {
            "id": "article-e0898924",
            "type": "article",
            "title": "Gemini 4 Hydrogen",
            "summary": "",
            "url": "https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-hydrogen/",
            "source": "hackernews",
            "date": "2026-11-12",
            "trendingScore": 100
        },
        {
            "id": "topic-large-language-models",
            "type": "topic",
            "title": "Large Language Models",
            "summary": "Foundation models trained on massive text corpora that can generate and understand natural language.",
            "connectionCount": 31
        },
        {
            "id": "topic-ai-agents",
            "type": "topic",
            "title": "AI Agents",
            "summary": "Autonomous AI systems that can plan, use tools, and take actions to accomplish goals.",
            "connectionCount": 20
        },
        {
            "id": "topic-nlp",
            "type": "topic",
            "title": "NLP",
            "summary": "Natural Language Processing: AI techniques for understanding and generating human language.",
            "connectionCount": 40
        },
        {
            "id": "topic-fine-tuning",
            "type": "topic",
            "title": "Fine-tuning",
            "summary": "Adapting pre-trained models to specific tasks or domains.",
            "connectionCount": 7
        },
        {
            "id": "topic-ai-reasoning",
            "type": "topic",
            "title": "AI Reasoning",
            "summary": "Methods to improve logical reasoning, mathematical problem-solving, and multi-step thinking in AI systems.",
            "connectionCount": 8
        },
        {
            "id": "topic-rag",
            "type": "topic",
            "title": "RAG",
            "summary": "Retrieval-Augmented Generation: combining LLMs with external knowledge retrieval for more accurate responses.",
            "connectionCount": 7
        },
        {
            "id": "topic-model-efficiency",
            "type": "topic",
            "title": "Model Efficiency",
            "summary": "Techniques to reduce computational costs and improve inference speed of AI models.",
            "connectionCount": 3
        },
        {
            "id": "topic-reinforcement-learning",
            "type": "topic",
            "title": "Reinforcement Learning",
            "summary": "Training AI through rewards and penalties to learn optimal behaviors.",
            "connectionCount": 16
        },
        {
            "id": "topic-ai-safety",
            "type": "topic",
            "title": "AI Safety",
            "summary": "Research focused on making AI systems safe, aligned with human values, and beneficial.",
            "connectionCount": 3
        },
        {
            "id": "topic-computer-vision",
            "type": "topic",
            "title": "Computer Vision",
            "summary": "AI systems for understanding and processing visual information from images and video.",
            "connectionCount": 7
        },
        {
            "id": "topic-multimodal-ai",
            "type": "topic",
            "title": "Multimodal AI",
            "summary": "Systems that process and understand multiple types of input including text, images, audio, and video.",
            "connectionCount": 3
        },
        {
            "id": "topic-prompt-engineering",
            "type": "topic",
            "title": "Prompt Engineering",
            "summary": "Methods for crafting effective prompts to guide AI model behavior and outputs.",
            "connectionCount": 4
        },
        {
            "id": "topic-diffusion-models",
            "type": "topic",
            "title": "Diffusion Models",
            "summary": "Generative models that create content by iteratively denoising random noise into structured outputs.",
            "connectionCount": 2
        },
        {
            "id": "org-google",
            "type": "organization",
            "title": "Google",
            "summary": "Google - AI research and development.",
            "connectionCount": 6
        },
        {
            "id": "org-hugging-face",
            "type": "organization",
            "title": "Hugging Face",
            "summary": "Hugging Face - AI research and development.",
            "connectionCount": 2
        },
        {
            "id": "org-nvidia",
            "type": "organization",
            "title": "NVIDIA",
            "summary": "NVIDIA - AI research and development.",
            "connectionCount": 4
        },
        {
            "id": "org-openai",
            "type": "organization",
            "title": "OpenAI",
            "summary": "OpenAI - AI research and development.",
            "connectionCount": 2
        },
        {
            "id": "org-anthropic",
            "type": "organization",
            "title": "Anthropic",
            "summary": "Anthropic - AI research and development.",
            "connectionCount": 1
        },
        {
            "id": "org-mistral",
            "type": "organization",
            "title": "Mistral",
            "summary": "Mistral - AI research and development.",
            "connectionCount": 1
        },
        {
            "id": "model-llama",
            "type": "model",
            "title": "Llama",
            "summary": "Llama AI model.",
            "connectionCount": 2
        },
        {
            "id": "model-claude",
            "type": "model",
            "title": "Claude",
            "summary": "Claude AI model.",
            "connectionCount": 15
        },
        {
            "id": "model-gemini",
            "type": "model",
            "title": "Gemini",
            "summary": "Gemini AI model.",
            "connectionCount": 20
        },
        {
            "id": "model-chatgpt",
            "type": "model",
            "title": "ChatGPT",
            "summary": "ChatGPT AI model.",
            "connectionCount": 1
        },
        {
            "id": "model-mistral",
            "type": "model",
            "title": "Mistral",
            "summary": "Mistral AI model.",
            "connectionCount": 1
        },
        {
            "id": "model-grok",
            "type": "model",
            "title": "Grok",
            "summary": "Grok AI model.",
            "connectionCount": 1
        }
    ],
    "edges": [
        {
            "source": "article-789d923f",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-789d923f",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-910f0459",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-910f0459",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-910f0459",
            "target": "model-llama",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-f514627b",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-f514627b",
            "target": "topic-fine-tuning",
            "relationship": "COVERS"
        },
        {
            "source": "article-a9f5ac62",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-a9f5ac62",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-a9f5ac62",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-c8c59e04",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-294bd9f5",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-294bd9f5",
            "target": "topic-model-efficiency",
            "relationship": "COVERS"
        },
        {
            "source": "article-294bd9f5",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-b8f4a30d",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-b8f4a30d",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-5b03b485",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-5b03b485",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-436ad407",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-ee7f42a5",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-ee7f42a5",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-67c9b16f",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-67c9b16f",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-57603665",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-57603665",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-26a62639",
            "target": "topic-ai-safety",
            "relationship": "COVERS"
        },
        {
            "source": "article-26a62639",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-26a62639",
            "target": "topic-fine-tuning",
            "relationship": "COVERS"
        },
        {
            "source": "article-03c9d131",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-03c9d131",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-03c9d131",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-03c9d131",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-39259697",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-39259697",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-39259697",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-fb334930",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-fb334930",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-c04e6208",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-c04e6208",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-c04e6208",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-c8b83062",
            "target": "topic-multimodal-ai",
            "relationship": "COVERS"
        },
        {
            "source": "article-c8b83062",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-c8b83062",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-38a69ddf",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-38a69ddf",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-38a69ddf",
            "target": "topic-ai-safety",
            "relationship": "COVERS"
        },
        {
            "source": "article-38a69ddf",
            "target": "topic-prompt-engineering",
            "relationship": "COVERS"
        },
        {
            "source": "article-38a69ddf",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-7797f9ed",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-0ae9dd1d",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-0ae9dd1d",
            "target": "topic-prompt-engineering",
            "relationship": "COVERS"
        },
        {
            "source": "article-b1e46780",
            "target": "topic-diffusion-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-b1e46780",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-b1e46780",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-b1e46780",
            "target": "topic-fine-tuning",
            "relationship": "COVERS"
        },
        {
            "source": "article-ac203f62",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-231ae5e2",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-5108bce7",
            "target": "topic-model-efficiency",
            "relationship": "COVERS"
        },
        {
            "source": "article-5108bce7",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-5406eadd",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-b2969b32",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-b2969b32",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-4c2c6384",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-d9cee7a9",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-38dc97f6",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-38dc97f6",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-0020dd2a",
            "target": "topic-multimodal-ai",
            "relationship": "COVERS"
        },
        {
            "source": "article-0020dd2a",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-0020dd2a",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-0020dd2a",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-61241346",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-61241346",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-61241346",
            "target": "topic-fine-tuning",
            "relationship": "COVERS"
        },
        {
            "source": "article-e2252d53",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-cffc02a2",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-cffc02a2",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-6bc53ef6",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-6bc53ef6",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-4fb37acb",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-4fb37acb",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-a9618a54",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-a9618a54",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-9aa193d3",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-1e6da13c",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-70fb34e7",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-70fb34e7",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-32151157",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-32151157",
            "target": "topic-diffusion-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-32151157",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-32151157",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-a81b9f8f",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-a81b9f8f",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-15fdb974",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-56ecbf67",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-56ecbf67",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-56ecbf67",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-56ecbf67",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-9ccb449f",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-9ccb449f",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-9ccb449f",
            "target": "topic-prompt-engineering",
            "relationship": "COVERS"
        },
        {
            "source": "article-9ccb449f",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-9ccb449f",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-d6c99d69",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-d6c99d69",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-d6c99d69",
            "target": "topic-model-efficiency",
            "relationship": "COVERS"
        },
        {
            "source": "article-d6c99d69",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-abcf06f5",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-abcf06f5",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-abcf06f5",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-b4fbfe19",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-b4fbfe19",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-ba492d0f",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-ba492d0f",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-ba492d0f",
            "target": "topic-fine-tuning",
            "relationship": "COVERS"
        },
        {
            "source": "article-cf88cd4a",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-cf88cd4a",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-cf88cd4a",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-cf88cd4a",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-cf88cd4a",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-cf88cd4a",
            "target": "topic-fine-tuning",
            "relationship": "COVERS"
        },
        {
            "source": "article-ee6a189c",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-da69039a",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-da69039a",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-3987a872",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-3987a872",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-3987a872",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-3586fd5c",
            "target": "topic-multimodal-ai",
            "relationship": "COVERS"
        },
        {
            "source": "article-3586fd5c",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-d3a6593e",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-d3a6593e",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-8477a815",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-8477a815",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-0b514ad9",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-0b514ad9",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-0b514ad9",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-da8db646",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-da8db646",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-1f76189b",
            "target": "model-llama",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-f052c684",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-d3916f44",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-d3916f44",
            "target": "topic-fine-tuning",
            "relationship": "COVERS"
        },
        {
            "source": "article-4c5ef8a2",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-4c5ef8a2",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-992e8c41",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-992e8c41",
            "target": "org-hugging-face",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-de10773c",
            "target": "org-hugging-face",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-0f553794",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-00fbeb28",
            "target": "org-nvidia",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-1127808f",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-9541439d",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-9541439d",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-9541439d",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-42d6602d",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-83f52ff8",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-bad9b100",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-bad9b100",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-756341d4",
            "target": "org-nvidia",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-dfff9e74",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-dfff9e74",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-0c8877bb",
            "target": "org-nvidia",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-a1df07d5",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-a1df07d5",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-5f4b98ca",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-a3e3f29b",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-a3e3f29b",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-0ec0016f",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-0ec0016f",
            "target": "topic-ai-safety",
            "relationship": "COVERS"
        },
        {
            "source": "article-e1a45ad6",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-1ac99517",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-8cd2858e",
            "target": "org-openai",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-8cd2858e",
            "target": "org-anthropic",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-efc0a0db",
            "target": "org-nvidia",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-fc3aa97d",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-7aee8ee7",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-cc2d4e61",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-f009549c",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-90b9361a",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-90b9361a",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-7918eeb1",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-7918eeb1",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-7918eeb1",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-6270073a",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-60661990",
            "target": "topic-prompt-engineering",
            "relationship": "COVERS"
        },
        {
            "source": "article-60661990",
            "target": "model-chatgpt",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-60661990",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-d9e633e6",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-90c609f8",
            "target": "org-mistral",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-90c609f8",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-90c609f8",
            "target": "model-mistral",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-7eef33e3",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-86468cda",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-76f26272",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-2eb502da",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-2eb502da",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-2eb502da",
            "target": "model-grok",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-7a5bff98",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-aea374a9",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-aea374a9",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-2925fc76",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-8ae64a0d",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-9ea4c258",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-b3f93f30",
            "target": "org-openai",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-b3f93f30",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-250664fb",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-9afe5c76",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-a6377243",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-e0898924",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "topic-large-language-models",
            "target": "topic-ai-reasoning",
            "relationship": "RELATED_TO"
        },
        {
            "source": "topic-large-language-models",
            "target": "topic-ai-agents",
            "relationship": "RELATED_TO"
        },
        {
            "source": "topic-large-language-models",
            "target": "topic-rag",
            "relationship": "RELATED_TO"
        },
        {
            "source": "topic-multimodal-ai",
            "target": "topic-computer-vision",
            "relationship": "RELATED_TO"
        },
        {
            "source": "topic-ai-agents",
            "target": "topic-prompt-engineering",
            "relationship": "RELATED_TO"
        },
        {
            "source": "topic-model-efficiency",
            "target": "topic-large-language-models",
            "relationship": "RELATED_TO"
        },
        {
            "source": "topic-ai-safety",
            "target": "topic-large-language-models",
            "relationship": "RELATED_TO"
        }
    ]
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AIChronicleData;
}
