// ================================================
// THE AI CHRONICLE - Knowledge Graph Data
// Auto-generated and updated daily via GitHub Actions
// Last updated: 2026-09-10
// ================================================

const AIChronicleData = {
    "metadata": {
        "lastUpdated": "2026-09-10T10:37:50.279325Z",
        "totalArticles": 133,
        "totalNodes": 154,
        "totalEdges": 205,
        "dateRange": {
            "start": "2026-09-03",
            "end": "2026-09-10"
        }
    },
    "nodes": [
        {
            "id": "article-8b25c51e",
            "type": "article",
            "title": "Beyond Right and Wrong: Evaluating Second-order Social Reasoning in Large Language Models",
            "summary": "arXiv:2609.05437v1 Announce Type: new Abstract: Previous AI alignment efforts have focused primarily on first-order social norms -- teaching models what is socially acceptable or unacceptable (e.g., `do not steal'). However, social intelligence depends not only on norm recognition, but also on anticipating who will enforce it and how (e.g., public shame or even imprisonment). These second-order expectations, known as metanorms, govern how people respond when social rules are broken. We introduce",
            "url": "https://arxiv.org/abs/2609.05437",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-e23485c0",
            "type": "article",
            "title": "CriticGen: Generation-Aware Evaluation as Actionable Feedback",
            "summary": "arXiv:2609.05439v1 Announce Type: new Abstract: Current evaluation methods for large language models are coarse-grained and decoupled from generation, producing generic explanations that fail to provide actionable feedback for model improvement. We propose CriticGen, a fine-grained, generation-aware evaluation framework that turns evaluation into actionable control for answer improvement. CriticGen first generates sample-specific evaluation dimensions and scoring criteria under high-level catego",
            "url": "https://arxiv.org/abs/2609.05439",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-cb56f2d0",
            "type": "article",
            "title": "When Does Memory Help? A Cost-Aware Evaluation of Long-Term Memory in Tool-Using LLM Agents",
            "summary": "arXiv:2609.05441v1 Announce Type: new Abstract: Long-term memory for LLM agents is evaluated today by conversational recall benchmarks (LoCoMo, LongMemEval), which measure question answering over dialogue history, not whether remembered facts change what a tool-using agent does. We present MERIT (Memory Evaluation for Realistic Instrumented Tasks), a benchmark and harness that measures the marginal utility of memory for task-executing agents under explicit cost accounting. MERIT provides episodi",
            "url": "https://arxiv.org/abs/2609.05441",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-7768e631",
            "type": "article",
            "title": "AutoFyn Technical Report: Non-Parametric Expert Iteration for Long-Horizon Agents",
            "summary": "arXiv:2609.05446v1 Announce Type: new Abstract: We introduce AutoFyn, an agent harness inspired by the Expert Iteration algorithm, adapting a frozen model across many rounds by updating persistent state from verified reward signals rather than model weights. Each round begins from a fresh model session, and durable information is reintroduced only through explicit interfaces such as persistent memory files, reports, and repository state. Within a round, an orchestrator explores, plans and builds",
            "url": "https://arxiv.org/abs/2609.05446",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-40be5fe8",
            "type": "article",
            "title": "Damage-Aware Bandit Pruning for Vision and Language Transformers",
            "summary": "arXiv:2609.05448v1 Announce Type: new Abstract: Structured post-training pruning of transformers requires selecting complete functional units whose suppression causes limited degradation. We formulate structured-unit selection for language and vision transformers as a damage-aware multi-armed bandit problem under a fixed candidate-evaluation budget. Attention heads and MLP channel groups are temporarily masked on calibration batches. Paired damage is the masked loss minus the base loss on the sa",
            "url": "https://arxiv.org/abs/2609.05448",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-f861f5c8",
            "type": "article",
            "title": "Compiling VGDL into Causal Models",
            "summary": "arXiv:2609.05459v1 Announce Type: new Abstract: Reinforcement learning and large language models often struggle to accurately capture the causal mechanics of game environments. Standard reinforcement learning agents tend to rely on spurious correlations, while large language models are prone to hallucinating game rules. Although causal reinforcement learning improves interpretability, there is currently no formal methodology to map complex game mechanics directly into causal models. To address t",
            "url": "https://arxiv.org/abs/2609.05459",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-5829b12a",
            "type": "article",
            "title": "ARC-Bench: Closed-Loop Replanning Masks Broken Action Ranking in Frozen JEPA World Models",
            "summary": "arXiv:2609.05461v1 Announce Type: new Abstract: Reward-free latent world models plan by scoring candidate actions with distances in a frozen latent space: an action is preferred if its predicted future embedding lands closer to the goal embedding. This silently assumes that latent closeness is action-rankable, i.e., that ordering candidates by latent distance agrees with ordering them by true cost. We audit this assumption directly. We introduce ARC-Bench, a no-leak, fixed-candidate protocol tha",
            "url": "https://arxiv.org/abs/2609.05461",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-ec956445",
            "type": "article",
            "title": "RAPID: Reliability-Aware Pair Importance Distillation",
            "summary": "arXiv:2609.05481v1 Announce Type: new Abstract: Inter example relational distillation transfers a teacher's representation geometry by matching relations among examples within a mini batch. Computing all pairs has quadratic complexity in the batch size, whereas uniform subsampling may use a limited relation budget inefficiently. We introduce Reliability Aware Pair Importance Distillation, or RAPID, which separates a reliability gated relational target from a full support adaptive pair proposal. ",
            "url": "https://arxiv.org/abs/2609.05481",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-8c9b237c",
            "type": "article",
            "title": "PGP-Clinical-TimeKAN: Prior-Guided Joint Probabilistic Forecasting of Clinical Trajectories",
            "summary": "arXiv:2609.05488v1 Announce Type: new Abstract: Clinical deterioration unfolds through coupled, partially observed trajectories, not a single diagnostic label. We introduce PGP-Clinical-TimeKAN, a trajectory-first framework for joint probabilistic forecasting of multivariate physiology. It combines missingness-aware temporal encoders, a soft organ-system prior, patient-specific relations, nonlinear Kolmogorov-Arnold messages, and a low-rank multivariate Student-t head. We evaluate 24-hour histor",
            "url": "https://arxiv.org/abs/2609.05488",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-99b54e34",
            "type": "article",
            "title": "SciLitBench: Benchmark and Design Principles for LLM-Powered Systematic Literature Reviews",
            "summary": "arXiv:2609.05505v1 Announce Type: new Abstract: Systematic reviews require sustained human judgment across thousands of records, yet existing evaluations of large language models (LLMs) typically examine review stages in isolation. We introduce SciLitBench, a multi-stage benchmark spanning title and abstract screening, full-text screening, and schema-guided data extraction, with 42,981 retrieved records, 1,012 full texts, and annotations for 888 included papers. Across 22 open-weight LLMs from s",
            "url": "https://arxiv.org/abs/2609.05505",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-390753c0",
            "type": "article",
            "title": "SCAFFOLD: Self-Improving Web Agents via Recursive Parametric Skill Abstraction",
            "summary": "arXiv:2609.05511v1 Announce Type: new Abstract: Web agents need to navigate visually rich, long-horizon interfaces that change across sites, yet most previous agents still learn each task in isolation and discard the procedural knowledge they accumulate. Recent skill-augmented frameworks take an important first step, but they treat the skill library as a flat or two-tier prompt-side cache and offer no principled mechanism for compressing redundancy or composing skills recursively. We introduce \\",
            "url": "https://arxiv.org/abs/2609.05511",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-34398729",
            "type": "article",
            "title": "Reasoning-Aware Compression: Identifying and Protecting Vulnerable Reasoning Circuits for Energy-Efficient LLM Deployment",
            "summary": "arXiv:2609.05512v1 Announce Type: new Abstract: Large Reasoning Models (LRMs) impose substantial energy costs during deployment, yet current compression methods apply uniform quantization across all components, risking damage to critical reasoning circuits. We present a reasoning-aware compression framework that benchmarks quantization conditions across five reasoning benchmarks, GSM8K, FOLIO, MATH-500, ProofWriter, and MuSiQue, with hardware-level GPU energy measurement; profiles per-module INT",
            "url": "https://arxiv.org/abs/2609.05512",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-63ad0e1f",
            "type": "article",
            "title": "When and What to Teach: Budget-Aware Online Adaptation for Web Agents",
            "summary": "arXiv:2609.05513v1 Announce Type: new Abstract: Web agents have achieved significant success in automating complex internet tasks but deploying them in real-world environments requires continuous online adaptation. Given that deploying powerful proprietary models remains commercially cost-prohibitive, practitioners must rely on lightweight local models that evolve post-deployment via online teaching from a stronger teacher. However, standard interactive feedback imposes prohibitive costs. We sho",
            "url": "https://arxiv.org/abs/2609.05513",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-eda0e237",
            "type": "article",
            "title": "The Failure Happens Before the Drift: The Social Dynamics of Values in LLM Agent Societies",
            "summary": "arXiv:2609.05514v1 Announce Type: new Abstract: Large Language Model (LLM)-based agents are increasingly used as proxies for human participants in social science research, yet it remains unclear whether they can faithfully simulate diverse and conflicting human value systems. We present a World Values Survey (WVS)-grounded simulation framework where culturally diverse agents with different communication styles engage in longitudinal, value-laden discussions. Across approximately 4,000 conversati",
            "url": "https://arxiv.org/abs/2609.05514",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-88e4e50e",
            "type": "article",
            "title": "Beyond \"AI Helps Humans\": Decision-Targeted Evaluation Design for Human-Agent Teams in the Agentic Era",
            "summary": "arXiv:2609.05527v1 Announce Type: new Abstract: Wherever a coding agent works under engineer supervision, or a clinical model assists a radiologist, the deployment question is whether to keep the human-AI workflow or replace it with the human alone or the agent alone. The human-AI workflow is worth keeping only if it beats both of those alternatives. Yet once it is deployed, neither alternative outcome is observed: recovering one means replaying the task under that alternative, and every replay ",
            "url": "https://arxiv.org/abs/2609.05527",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-9d857600",
            "type": "article",
            "title": "EdgeMem: LLM-Free Agent Memory Construction and Retrieval via Evidence-Preserving Multi-Anchor Hypergraph",
            "summary": "arXiv:2609.05553v1 Announce Type: new Abstract: Agent memory allows LLM agents to use earlier interactions when answering new queries. Existing methods often compress interaction histories into summaries or other LLM-generated representations. Repeated generation adds cost and can discard answer-bearing details before the system knows what a future query will require. We propose EdgeMem, an agent-memory method built around a simple principle: preserve original interaction turns and organize them",
            "url": "https://arxiv.org/abs/2609.05553",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-2f767d48",
            "type": "article",
            "title": "Deep belief networks are exact",
            "summary": "arXiv:2609.05572v1 Announce Type: new Abstract: We prove that every strictly positive probability distribution on \\(\\{-1,1\\}^n\\) is represented exactly by a sigmoid belief network with finite parameters. This answers a question of Sutskever and Hinton. The proof upgrades their probability-sharing approximation to exact representation using Brouwer's fixed-point theorem.",
            "url": "https://arxiv.org/abs/2609.05572",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-8ad8a572",
            "type": "article",
            "title": "EnvCraft: Synthesizing Executable Environments in Agentic RL for Claw-like Agent",
            "summary": "arXiv:2609.05576v1 Announce Type: new Abstract: The paradigm of LLMs has rapidly shifted from passive language interfaces to autonomous Claw-like agents that execute long-horizon tasks across stateful workspaces. While Agentic Reinforcement Learning (Agentic RL) provides a promising path to optimize these agents, its scaling is heavily bottlenecked by the severe scarcity of interactive training environments. Existing synthetic environments are strictly limited to tool-calling endpoints, renderin",
            "url": "https://arxiv.org/abs/2609.05576",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-386b820e",
            "type": "article",
            "title": "Planning and Scheduling Business Processes under Control-Flow Uncertainty",
            "summary": "arXiv:2609.05578v1 Announce Type: new Abstract: Scheduling activities in business processes can improve efficiency (e.g., reduce makespan), but is challenging because the exact sequence of activities required to complete a case is often uncertain due to decisions based on data that emerges during execution. Nevertheless, probabilistic information regarding such decisions can often be estimated or derived from historical execution logs, and can help anticipate which execution paths are likely to ",
            "url": "https://arxiv.org/abs/2609.05578",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-320cc556",
            "type": "article",
            "title": "Agents Trust Tools Too Much: Measuring Reliance on Unreliable Tools",
            "summary": "arXiv:2609.05587v1 Announce Type: new Abstract: Existing evaluations of tool-using agents primarily measure whether an agent can successfully complete diverse tasks with tools. These evaluations generally assume that tools return reliable information. However, tool returns in real-world systems can be plausible yet incorrect. We investigate how agents respond to unreliable tool returns by evaluating fourteen LLMs using three tools-web search, LLM sub-agent delegation, and code execution. For eac",
            "url": "https://arxiv.org/abs/2609.05587",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-269f1f39",
            "type": "article",
            "title": "AhaBench: Do Agents Learn from Prior Experience? A Benchmark for Long-Horizon Continual Learning",
            "summary": "arXiv:2609.05435v1 Announce Type: new Abstract: Modern language agents are expected to operate over long horizons: they ask follow-up questions, reuse worked examples, handle tool feedback, and adapt to delayed consequences. Most evaluations still reset the agent after a prompt or score only the final state of one trajectory. AhaBench asks a more operational question: when a fixed model receives useful experience, does its later behavior improve under a related evaluation condition where the obv",
            "url": "https://arxiv.org/abs/2609.05435",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-de919604",
            "type": "article",
            "title": "When Do Options Help? Policy Necrosis and Redundant Coverage in Option-Critic",
            "summary": "arXiv:2609.05508v1 Announce Type: new Abstract: Option-critic learns options: sub-policies together with a learned rule for when each one hands control back. Its headline result is that performance improves as options are added. We explain that result, with theory and experiment. First, the termination rule option-critic learns by maximising return contributes nothing. When the termination test and the policy that picks options read the same values, the test fires at every step, so the learned r",
            "url": "https://arxiv.org/abs/2609.05508",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-5b8248bb",
            "type": "article",
            "title": "Multi-granularity Adaptive Hypergraph Representation Learning via Granular-ball",
            "summary": "arXiv:2609.05574v1 Announce Type: new Abstract: Hypergraph representation learning aims to capture high-order information in graphs by constructing hyperedges that simultaneously connect multiple nodes. These hyperedges adapt to the graph's topological features, facilitating the extraction of high-order relationships at multiple granularities. Most prior work relies on predefined definitions to generate hyperedges, overlooking the diversity in graph topological structures and the multi-granulari",
            "url": "https://arxiv.org/abs/2609.05574",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-0045cbdc",
            "type": "article",
            "title": "Capsule Lens: Locating and Tracking Concept Geometry in Model Representations",
            "summary": "arXiv:2609.05575v1 Announce Type: new Abstract: Understanding how concepts are encoded in the internal representations of machine learning models is a central problem in mechanistic interpretability, essential both for the science of deep learning and for the trustworthy deployment of increasingly capable models. Existing approaches to interpret model representations mainly map representations onto more interpretable spaces and do not directly characterize how concepts occupy representation spac",
            "url": "https://arxiv.org/abs/2609.05575",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-2326a9cb",
            "type": "article",
            "title": "HB-PVI: A Hierarchical Bayesian Personalization and Value-of-Information Framework for Complex Activity Recognition",
            "summary": "arXiv:2609.05582v1 Announce Type: new Abstract: Personalization can improve activity-recognition performance, but participant-specific gains are heterogeneous, and every additional calibration label has an acquisition cost. This study presents HB-PVI, a hierarchical Bayesian personalization and value-of-information framework jointly modeling participant heterogeneity, the benefit and harm of four personalization mechanisms, and the economic value of an additional label, for the 47-participant MU",
            "url": "https://arxiv.org/abs/2609.05582",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-1e913121",
            "type": "article",
            "title": "Endogenous Exploration in Reinforcement Learning with Intrinsic Curiosity",
            "summary": "arXiv:2609.05650v1 Announce Type: new Abstract: We propose a reinforcement learning framework in which exploration is driven by intrinsic curiosity, designed for scenarios where environments are non-stationary and rewards are sparse, delayed, uninformative, or absent. In our model, action selection is guided by a combination of external rewards and an epistemic motivation mechanism that biases the agent toward structured exploratory directions. The central hypothesis is that effective exploratio",
            "url": "https://arxiv.org/abs/2609.05650",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-a843ad46",
            "type": "article",
            "title": "Robustness of LLM-Generated SystemVerilog Assertions to Semantics-Preserving RTL Transformations",
            "summary": "arXiv:2609.05658v1 Announce Type: new Abstract: Large language models (LLMs) are increasingly being explored for automating SystemVerilog Assertion (SVA) generation, yet most evaluations report correctness on a single syntactic representation of an input. Such point accuracy does not reveal whether a model's correct output is stable when the same RTL behavior is written differently. This paper presents a controlled metamorphic evaluation of LLM-based SVA generation under semantics-preserving RTL",
            "url": "https://arxiv.org/abs/2609.05658",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-7ad1ccf6",
            "type": "article",
            "title": "PAC-Private Autoregressive Generation: Calibrating Noise to Ensemble Disagreement",
            "summary": "arXiv:2609.05676v1 Announce Type: new Abstract: Language models adapted on private text are often served through APIs, so privacy leakage occurs through generated outputs rather than exposed weights. Private prediction protects these releases. Methods such as PMixED incur privacy cost at each release and increasingly rely on the public model over long horizons. PAC privacy instead calibrates noise to output variability across possible secrets, adding less noise when predictions are stable. To ou",
            "url": "https://arxiv.org/abs/2609.05676",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-6771e1ef",
            "type": "article",
            "title": "Connecting Score Matching, Maximum Likelihood, and Expectation-Maximization in Mixed Linear Regression",
            "summary": "arXiv:2609.05688v1 Announce Type: new Abstract: We study variance-preserving diffusion of the response in mixed linear regression (MLR) with unknown mixing weights. Our analysis separates the statistical guarantees of score matching from the loss geometry and optimization signal at a fixed diffusion noise level. The KL divergence links the denoising score matching objective integrated over the diffusion path with the likelihood and a terminal discrepancy. Under mild regularity conditions and ter",
            "url": "https://arxiv.org/abs/2609.05688",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-5b88a2c1",
            "type": "article",
            "title": "GraphNOSE: A Graph Transformer in Olfaction",
            "summary": "arXiv:2609.05694v1 Announce Type: new Abstract: Predicting olfactory qualities from molecular structure is an open problem in chemoinformatics. Although linear models can link molecular features to odor descriptors, they often fail when extrapolating to novel chemical scaffolds, extreme molecular weights, or complex odor mixtures. To address this, we introduce GraphNOSE, an open-source graph transformer framework that predicts multi-label odor descriptors from simplified molecular-input line-ent",
            "url": "https://arxiv.org/abs/2609.05694",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-b9e6ee3d",
            "type": "article",
            "title": "Analysis of Respiratory Sinus Arrhythmia with Neural Networks",
            "summary": "arXiv:2609.05698v1 Announce Type: new Abstract: The paper introduces a neural network-based approach for analyzing ECG signals to estimate respiratory rate by leveraging the phe- nomenon of Respiratory Sinus Arrhythmia (RSA). Our method employs a deep learning model trained to predict respiratory waveforms directly from ECG input data. To achieve this, we developed and evaluated three different neural network architectures capable of automatically extract- ing relevant features from ECG signals ",
            "url": "https://arxiv.org/abs/2609.05698",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-6156280c",
            "type": "article",
            "title": "Newton Matching for Generative Modeling: A Unified Framework for Fine-Tuning and Sampling",
            "summary": "arXiv:2609.05727v1 Announce Type: new Abstract: We develop Newton Matching, a unified framework for fine-tuning and sampling in generative modeling. The target is $\\pi\\propto\\mu e^{\\tau r}$, where $r$ is the reward, $\\tau>0$ the inverse temperature, and $\\mu$ denotes the pretrained model's terminal density for fine-tuning or the constant $1$ for sampling. We shift the paradigm from isolated losses to iterative optimization over canonical models: population minimizers of standard conditional matc",
            "url": "https://arxiv.org/abs/2609.05727",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-6ce925d4",
            "type": "article",
            "title": "A Multi-Source Ensemble Approach to Candidate Generation for Alternative Vacation Rental Property Recommendations",
            "summary": "arXiv:2609.05748v1 Announce Type: new Abstract: Alternative property recommendations play a critical role in vacation rental marketplaces, helping users discover relevant options when viewing a specific listing. However, generating high-quality candidate alternatives presents unique challenges: heterogeneous inventory, geographic constraints, rapid availability changes, and long-tail property distributions. We present a comprehensive study of candidate generation (CG) approaches for vacation ren",
            "url": "https://arxiv.org/abs/2609.05748",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-d9add56a",
            "type": "article",
            "title": "Data Scout: Targeted Web Crawling for Domain-Specific Pretraining Corpora",
            "summary": "arXiv:2609.05766v1 Announce Type: new Abstract: The dominant approach to building domain-specific pretraining corpora is to filter large web archives such as CommonCrawl. This works well for popular domains but breaks down for specialized ones, where relevant content is sparse and often beyond the reach of popularity-driven crawlers. We present Data Scout, which inverts this: instead of filtering an archive, it directs a targeted crawl. An LLM expands a root topic into a taxonomy and thousands o",
            "url": "https://arxiv.org/abs/2609.05766",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-70d4862f",
            "type": "article",
            "title": "RAPTOR: Role-Aware Private Training for Mixture-of-Experts",
            "summary": "arXiv:2609.05770v1 Announce Type: new Abstract: Differentially private (DP) fine-tuning methods treat sparse Mixture-of-Experts (MoE) models as a single dense block, ignoring that shared layers see all data while experts only see routed records. We identify and formally characterize three resulting failure modes: global clipping suppresses expert gradients, batch-level normalization dilutes sparse expert updates, and fixed privacy noise degrades signal-to-noise ratio on low-load experts. We intr",
            "url": "https://arxiv.org/abs/2609.05770",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-1a66910b",
            "type": "article",
            "title": "Nonlinear elliptic homogenization with the parametric Deep Ritz method",
            "summary": "arXiv:2609.05778v1 Announce Type: new Abstract: Elliptic homogenization is used to determine coarse-grained properties of materials with features on small scales. When these small scale features have rapid, periodic fluctuations, the solution field corresponding to a homogenized constitutive relation closely resembles the true solution based on the heterogeneous material. This homogenized behavior of the material is computed from a cell problem, where a cell is defined to be one period of the fl",
            "url": "https://arxiv.org/abs/2609.05778",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-7da30aa3",
            "type": "article",
            "title": "Online Learning with LLM Experts from Limited Feedback",
            "summary": "arXiv:2609.05820v1 Announce Type: new Abstract: We study adaptive routing of prompts to large language model (LLM) experts to maximize response quality in an online setting with limited feedback. We formulate it as a bandit problem with $K$ actions that represent experts and $d$ features that encode prompts, over a horizon of $T$ rounds. We propose algorithms that strategically select and observe rewards to minimize regret. In the full-information setting, we achieve a regret of $\\tilde{O}(d T /",
            "url": "https://arxiv.org/abs/2609.05820",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-67257591",
            "type": "article",
            "title": "Generalizing HVAC Control With Domain Randomized Reinforcement Learning",
            "summary": "arXiv:2609.05822v1 Announce Type: new Abstract: Deploying advanced HVAC (Heating, Ventilation and Air Conditioning) controllers at scale remains difficult because performance often depends on accurate building models or per-site retuning. We propose NOMAD-RL (Neural Online Meta-Adaptation for Dynamics), a general-purpose Reinforcement Learning (RL) controller designed to transfer across heterogeneous thermal zones through a universal, non-invasive thermostat interface. The controller acts on tem",
            "url": "https://arxiv.org/abs/2609.05822",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-dca4286d",
            "type": "article",
            "title": "Scaling Optimal Classification Trees via Adaptive Feature and Sample Reduction",
            "summary": "arXiv:2609.05826v1 Announce Type: new Abstract: Dynamic programming for optimal classification trees becomes computationally expensive as the numbers of features and training samples increase. We develop a joint feature- and sample-space reduction framework based on STreeD. Weighted STreeD merges duplicate records created after projection onto a fixed candidate set into weighted representatives. This reduces sample-dependent computation without changing the fixed-candidate optimization problem. ",
            "url": "https://arxiv.org/abs/2609.05826",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-f992524a",
            "type": "article",
            "title": "SAFEGuard: Detect Optimization-Based Jailbreak Attacks Through Harmful Semantic Analysis and Fluency Measurement",
            "summary": "arXiv:2609.05850v1 Announce Type: new Abstract: Despite the significant efforts devoted to aligning large language models (LLMs) with human values and ensuring safe deployment, recent work has revealed that LLMs remain vulnerable to adversarial jailbreak attacks that can bypass safety guardrails and elicit harmful responses. Many defense methods are proposed to detect jailbreaks but they are limited in their effectiveness to counter wide-range optimization-based jailbreak mechanisms that can yie",
            "url": "https://arxiv.org/abs/2609.05850",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-f3ce0512",
            "type": "article",
            "title": "X-CoSD: Communication-Efficient Cross-Vocabulary Collaborative Speculative Decoding",
            "summary": "arXiv:2609.09166v1 Announce Type: new Abstract: This paper investigates collaborative speculative decoding (CoSD), a distributed large language model (LLM) inference framework in which an on-device small language model (SLM) drafts candidate tokens and a server LLM verifies them. Existing CoSD methods assume a shared vocabulary between the SLM and the LLM and incur substantial communication load because residual resampling requires token distribution exchange between the user device and the edge",
            "url": "https://arxiv.org/abs/2609.09166",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-334d3757",
            "type": "article",
            "title": "StochBench: A Domain-Specific Benchmark for Stochastic Processes in Lean",
            "summary": "arXiv:2609.09264v1 Announce Type: new Abstract: Leading benchmarks for formal theorem proving with large language models are small collections drawn from competition math, such as the IMO and Putnam, that poorly represent field-specific applications. We introduce StochBench, a Lean 4 benchmark of 450 graduate stochastic-processes problems at varying abstraction levels, each paired with its natural-language source. Addressing a field underrepresented in Mathlib, it covers finite and countable Mar",
            "url": "https://arxiv.org/abs/2609.09264",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-d8e070b6",
            "type": "article",
            "title": "Osprey: Target-agnostic Pre-training Makes Stronger Drafters in Speculative Decoding",
            "summary": "arXiv:2609.09338v1 Announce Type: new Abstract: Speculative decoding is critical for accelerating LLM inference. However, the speedup is fragile: drafters are typically trained against a narrow distribution for a single target model, and their acceptance rate collapses under workload shifts. This is a striking inversion of modern LLM development, where target models are valued precisely for the broad generalization they acquire through large-scale pretraining. We argue that the natural remedy, p",
            "url": "https://arxiv.org/abs/2609.09338",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-4cd9b4ff",
            "type": "article",
            "title": "SWORD: Wikidata-based Distortions Reveal Hidden Cross-Lingual Inconsistencies in LLM Factual Error Rejection",
            "summary": "arXiv:2609.09349v1 Announce Type: new Abstract: Modern LLMs demonstrate impressive multilingual performance, yet standard benchmarks primarily reward selecting correct answers rather than evaluating genuine factual understanding. We introduce Systematic Wikidata-based Object-Relation Distortion (SWORD), a benchmark that evaluates whether models consistently reject factual errors across languages. SWORD generates syntactically well-formed but factually incorrect statements in eight widely spoken ",
            "url": "https://arxiv.org/abs/2609.09349",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-98a5601a",
            "type": "article",
            "title": "Auditable Emergency Triage for Maternal and Newborn Care in India",
            "summary": "arXiv:2609.09356v1 Announce Type: new Abstract: At Noora Health, our nurses answer more than 50,000 medical queries per month on our WhatsApp-based service that provides caregivers with on-demand support. Their most time-critical task is emergency triage: deciding which queries need immediate in-person attention. To support them, we built a system that uses a large language model (LLM) to classify whether a message is an emergency and provide a rationale for interpretability. But the system was ",
            "url": "https://arxiv.org/abs/2609.09356",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-6702cf8a",
            "type": "article",
            "title": "Do LLMs Make More Mistakes If They Do Not Believe the Input Data?",
            "summary": "arXiv:2609.09363v1 Announce Type: new Abstract: Large language models (LLMs) are prone to hallucinating or misinterpreting facts, which impairs their usability in retrieval-augmented generation or data-to-text systems. We analyse how faithfulness of LLMs to provided context depends on how plausible they perceive the context to be (context-memory conflict). To better identify error patterns, we make use of the increased difficulty of non-English and low-resource language text generation and input",
            "url": "https://arxiv.org/abs/2609.09363",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-52c2df42",
            "type": "article",
            "title": "Benchmarking Hybrid Deep Research Across Database Querying and Web Search",
            "summary": "arXiv:2609.09410v1 Announce Type: new Abstract: While autonomous agents have made significant strides in \"deep research\" by iteratively navigating the open web to synthesize information, real-world problem-solving is rarely confined to a single environment. Complex analytical tasks inherently require agents to weave together evidence from both ambiguous unstructured text (e.g., the open web) and highly precise structured data (e.g., relational databases). However, existing benchmarks evaluate th",
            "url": "https://arxiv.org/abs/2609.09410",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-aa2fe68f",
            "type": "article",
            "title": "Edu-QuRating: Multi-Dimensional Educational Data Curation with Distilled Pairwise Judgements",
            "summary": "arXiv:2609.09425v1 Announce Type: new Abstract: Educational data filters have become a practical way to improve language-model pre-training, but most filters treat educational value as a single scalar property. This may be too broad for some applications, especially if the data set already features a high density of educational material. Useful learning material needs to be accurate, engaging, well structured, and appropriate for the intended audience and application (e.g. learner- vs teacher-fa",
            "url": "https://arxiv.org/abs/2609.09425",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-21a4ea1d",
            "type": "article",
            "title": "The Mutations of Machine Speech",
            "summary": "arXiv:2609.09496v1 Announce Type: new Abstract: Algorithmic outputs now populate the digital environments through which contemporary life is organized. The role of law in facilitating and constituting (rather than merely responding to) these processes is gaining increasing traction across scholarly accounts. This inquiry traces the evolution of algorithmic outputs attending to their legal underpinnings and social implications, surfacing the mutations of machine speech. The first mutation redefin",
            "url": "https://arxiv.org/abs/2609.09496",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-3a013d6c",
            "type": "article",
            "title": "TEFM: Token-Efficient Faithful Modeling for Structured Data",
            "summary": "arXiv:2609.09552v1 Announce Type: new Abstract: In this paper, we solve two fundamental obstacles in applying LLMs to critical domains: token efficiency and faithfulness. To address both constraints jointly, we present TEFM (Token-Efficient Faithful Modeling), a framework designed for structured data analysis in critical domains. TEFM achieves token efficiency by compressing lengthy structured observations into compact Behavioral Code tokens, dramatically reducing token consumption with minimal ",
            "url": "https://arxiv.org/abs/2609.09552",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-c4703e85",
            "type": "article",
            "title": "BuzzASR: A Swarm of 100+ Monolingual Speech Recognition Models",
            "summary": "arXiv:2609.09554v1 Announce Type: new Abstract: We introduce BuzzASR, a collection of language-specialized fine-tuned Whisper models adapted for automatic speech recognition (ASR) in 102 languages. Large end-to-end Transformer-based ASR models such as Whisper have revolutionized ASR, but most prominent models are highly multilingual. As a result, these models often perform poorly on languages less well-represented in their training set. While it has long been known that effective language adapta",
            "url": "https://arxiv.org/abs/2609.09554",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-2e8f68d6",
            "type": "article",
            "title": "Towards Automatic Evolution Tree Generation from Citation Graphs",
            "summary": "arXiv:2609.09561v1 Announce Type: new Abstract: Surveys remain the primary way researchers grasp the lineage of methods within an AI subfield, but they scale poorly against the current rate of publication. Existing taxonomy-induction methods are largely leaf-bound and time-agnostic; they tend to force transitional papers into mature leaves and can create topological inversions between ancestors and descendants. We propose EvoTree, a staged framework that decouples conceptual backbone learning fr",
            "url": "https://arxiv.org/abs/2609.09561",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-8fe51aaf",
            "type": "article",
            "title": "Reproducing Omitted Temporal Expressions in Japanese News for Retrieval-Augmented Applications",
            "summary": "arXiv:2609.09569v1 Announce Type: new Abstract: News articles often contain omitted temporal expressions, such as day-only or month-only mentions, which must be interpreted with reference to the publication date. When such articles are indexed or processed as standalone text in search and retrieval-augmented generation (RAG) systems, these omissions can cause temporal mismatches and unstable interpretation by large language models. We focus on reproducing omitted temporal expressions as concrete",
            "url": "https://arxiv.org/abs/2609.09569",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-ac5f43f0",
            "type": "article",
            "title": "Beyond Top Words: MonoTM for Topic Modeling with Interpretable Monosemantic Features",
            "summary": "arXiv:2609.09575v1 Announce Type: new Abstract: Topic models summarize large text corpora, but top-ranked words often provide only a limited representation of topic semantics. Sparse autoencoders (SAEs) offer a way to move beyond word-level descriptors by extracting interpretable features from dense representations, yet how feature interpretability relates to topic-inference quality remains unclear. We introduce \\textbf{MonoTM}, an interpretable topic modeling framework that decouples these role",
            "url": "https://arxiv.org/abs/2609.09575",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-434c41dd",
            "type": "article",
            "title": "SEA-SpeechBench: A Large-Scale Multitask Benchmark for Speech Understanding Across Southeast Asia",
            "summary": "arXiv:2609.09672v1 Announce Type: new Abstract: The rapid advancement of audio and multimodal large language models has unlocked transformative speech understanding capabilities, yet evaluation frameworks remain predominantly English-centric, leaving Southeast Asian (SEA) languages critically underrepresented. We introduce SEA-SpeechBench, to the best of our knowledge, the first large-scale multitask benchmark that evaluates speech understanding in 11 SEA languages through 97,194 samples across ",
            "url": "https://arxiv.org/abs/2609.09672",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-27cd6838",
            "type": "article",
            "title": "X2-NativeCursor: Native-Token Text Progress Tracking for Incremental-Text Streaming Codec TTS",
            "summary": "arXiv:2609.09677v1 Announce Type: new Abstract: Incremental-text streaming text-to-speech (TTS) needs online text progress tracking for synchronized highlighting, interruption handling, and dialogue-history updates. Input text arrives before it is spoken, so text arrival alone cannot indicate speech progress. Existing waveform-based alignment requires complete audio or adds acoustic processing during streaming. We propose X2-NativeCursor, a lightweight observer that tracks progress from native s",
            "url": "https://arxiv.org/abs/2609.09677",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-6b48ca84",
            "type": "article",
            "title": "Which Medical Questions Deserve Rationales? Perturbation-Sensitive Selection for Robust QA",
            "summary": "arXiv:2609.09684v1 Announce Type: new Abstract: Medical question-answering datasets often contain answer labels, whereas high-quality rationales remain scarce, noisy, or costly to validate. This changes the acquisition question: rather than asking which questions should be labeled, we ask which already-labeled questions should receive rationale supervision under a fixed token budget. We study an offline version of this problem in which candidate rationales are visible to the selector but withhel",
            "url": "https://arxiv.org/abs/2609.09684",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-ec5828eb",
            "type": "article",
            "title": "Looped GPT-BERT: Trading Parameters for Computation in Small Language Modeling",
            "summary": "arXiv:2609.09691v1 Announce Type: new Abstract: When training data are limited, increasing parameter count is not the only way to improve language-model performance. A small parameter set, when repeatedly applied, can also deliver comparable performance. We study Looped GPT-BERT in the BabyLM 2026 Strict-small setting, combining GPT-BERT's masked next-token and causal language-modeling objectives with depth-wise parameter sharing. We train on a preprocessed 7.48M-word English corpus and compare ",
            "url": "https://arxiv.org/abs/2609.09691",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-8c02b877",
            "type": "article",
            "title": "When Auditors Fabricate: Batch-Size Degradation and Confident Hallucination in LLM Detection of Planted Document Contamination",
            "summary": "arXiv:2609.09696v1 Announce Type: new Abstract: Large language models are increasingly proposed as automated auditors of document quality, yet their reliability as detectors of planted errors is poorly characterised. We construct a contaminated corpus of 150 academic papers spanning supply chain management and medical research, injecting 450 known contaminants of three types: typographical corruption, semantic reversal, and absurd out-of-context insertion. We then evaluate Google Gemini 3.0 Pro'",
            "url": "https://arxiv.org/abs/2609.09696",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-cd6b6799",
            "type": "article",
            "title": "Scaling E-Commerce Attribute Extraction with Parallel Decoding",
            "summary": "arXiv:2609.09716v1 Announce Type: new Abstract: Customers rely on specific product attributes to compare products and make purchasing decisions, but e-commerce catalogs are messy and unstructured, making it difficult to identify which attributes matter most and extract them at scale. Standard Attribute Value Extraction (AVE) systems treat all attributes equally, producing large, inconsistent attribute sets that do not reflect the factors consumers use to differentiate products. We introduce a tw",
            "url": "https://arxiv.org/abs/2609.09716",
            "source": "arxiv",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-43d2948c",
            "type": "article",
            "title": "Get ready for the game with new football features in Search",
            "summary": "An illustrated graphic set against a vibrant green background featuring American football elements, including a gold trophy, a blue helmet, a silver whistle, a football, a mini scoreboard, and play diagrams, with the icon for AI Mode in Google Search in t",
            "url": "https://blog.google/products-and-platforms/products/search/football-features-google-search/",
            "source": "blogs",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-32522e85",
            "type": "article",
            "title": "Recreating a 70-year love story frame by frame",
            "summary": "An elderly couple sitting in a movie theater. Overlayed are \"Teulluride Film Festival\" and \"Love, Rendered\"",
            "url": "https://blog.google/innovation-and-ai/technology/ai/love-rendered-film/",
            "source": "blogs",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-0e4c7024",
            "type": "article",
            "title": "IBM releases SOTA Granite Time Series PatchTST-FM-r2 model with commercial-friendly license",
            "summary": "",
            "url": "https://huggingface.co/blog/ibm-research/ibm-releases-sota-granite-time-series",
            "source": "blogs",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-c847825e",
            "type": "article",
            "title": "Safety for Whom? Refusing the Right Subset of a Topic, Not the Whole Topic",
            "summary": "",
            "url": "https://huggingface.co/blog/MultiverseComputingCAI/safety-for-whom",
            "source": "blogs",
            "date": "2026-09-08",
            "trendingScore": 50
        },
        {
            "id": "article-aad76210",
            "type": "article",
            "title": "NeoMME: an efficient Multimodal-native and Multilingual Encoder",
            "summary": "",
            "url": "https://huggingface.co/blog/Hcompany/neomme",
            "source": "blogs",
            "date": "2026-09-03",
            "trendingScore": 50
        },
        {
            "id": "article-388663d3",
            "type": "article",
            "title": "What OpenAI\u2019s latest controversy tells us about the future of math",
            "summary": "OpenAI\u2019s latest mathematical milestone has quickly become mired in controversy. Today, the company announced that its agents have solved one of the Millennium Prize Problems, some of the most important open problems in mathematics. Under normal circumstances, that solution would be a huge feather in OpenAI\u2019s cap. But the announcement has been overshadowed by accusations&#8230;",
            "url": "https://www.technologyreview.com/2026/09/08/1143747/what-openais-latest-controversy-tells-us-about-the-future-of-math/",
            "source": "blogs",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-f4a18652",
            "type": "article",
            "title": "This AI entrepreneur is developing agents that can plan ahead for the unexpected",
            "summary": "Danijar Hafner\u2019s office in San Francisco\u2019s SoMa district sits mostly empty. His brand-new startup is still in stealth mode and doesn\u2019t even have its name on the door. On the day I visit, there\u2019s only one other person there, and little in the way of furniture. But what it lacks in decor, it makes up&#8230;",
            "url": "https://www.technologyreview.com/2026/09/08/1142088/danijar-hafner-developing-plan-ahead-agents/",
            "source": "blogs",
            "date": "2026-09-08",
            "trendingScore": 50
        },
        {
            "id": "article-c5405304",
            "type": "article",
            "title": "Architecting memory and storage in the AI era",
            "summary": "The era of AI inference has arrived. Imagine a healthcare system analyzing millions of data points in real time to accelerate life-saving medical research, or an intelligent assistant instantly resolving thousands of complex customer needs at once. These real-world breakthroughs rely on advanced infrastructure acting as the engine of continuous intelligence, powering real-time services while&#8230;",
            "url": "https://www.technologyreview.com/2026/09/04/1140872/architecting-memory-and-storage-in-the-ai-era/",
            "source": "blogs",
            "date": "2026-09-04",
            "trendingScore": 50
        },
        {
            "id": "article-16db96c9",
            "type": "article",
            "title": "Data from drones in Ukraine is fueling a new Wild West marketplace",
            "summary": "Battlefields in Ukraine are littered with the remnants of drones, which are now firmly established as a critical weapon of modern warfare. But behind all that wreckage, there\u2019s a new gold mine for the defense sector. The data drones generate will far outlast the wars in which they are used to fight, increasingly becoming part&#8230;",
            "url": "https://www.technologyreview.com/2026/09/04/1143452/drone-data-wild-west/",
            "source": "blogs",
            "date": "2026-09-04",
            "trendingScore": 50
        },
        {
            "id": "article-b0b4d640",
            "type": "article",
            "title": "Gemini 3.8 Flash passed Claude Fable 5.1 on this bench",
            "summary": "",
            "url": "https://chessbench.ai",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-5bc17ec6",
            "type": "article",
            "title": "Nvidia and Palantir want to speed up the AI buildout. Nvidia is first in line",
            "summary": "",
            "url": "https://www.fastcompany.com/91604370/nvidia-palantir-sovereign-ai-supply-chains",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-de0a3185",
            "type": "article",
            "title": "Show HN: Community curated list of 310 AI providers offering $4.3k free credits",
            "summary": "",
            "url": "https://www.uprouter.online/",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-b646879c",
            "type": "article",
            "title": "Ask HN: As a teacher how should I teach kids to use AI?",
            "summary": "",
            "url": "https://news.ycombinator.com/item?id=49641100",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-b905f173",
            "type": "article",
            "title": "Laid-Off Developers Create AI Model to Replace CEOs and Other Executives",
            "summary": "",
            "url": "https://uk.pcmag.com/ai/167048/laid-off-developers-create-ai-model-to-replace-ceos-and-other-executives",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-fb236a78",
            "type": "article",
            "title": "Monitor AI visibility with n8n in five nodes",
            "summary": "",
            "url": "https://heeb.ai/blog/monitor-ai-visibility-with-n8n",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-0ab30f68",
            "type": "article",
            "title": "Show HN: Stroq \u2013 a firewall that knows why your AI agent ran that command",
            "summary": "",
            "url": "https://github.com/AGGIB/Stroq",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-1abdc22a",
            "type": "article",
            "title": "AI-Translation-Stack",
            "summary": "",
            "url": "https://github.com/antonihabek/AI-translation-stack",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-ee057021",
            "type": "article",
            "title": "Lawmakers blast AI companies after researcher warns of human extinction by 2030",
            "summary": "",
            "url": "https://www.theguardian.com/technology/2026/sep/09/lawmakers-blast-ai-human-extinct-2030",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-5b60bfe8",
            "type": "article",
            "title": "Call of Duty 2 running in Browser. Ported in 10 hours with GPT-6 Astra",
            "summary": "",
            "url": "https://www.reddit.com/r/aigamedev/comments/1wa11xy/call_of_duty_2_running_in_browser_ported_in_10/",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-448d938b",
            "type": "article",
            "title": "I Was Offered Money to Tell You AI Will Kill Us [video]",
            "summary": "",
            "url": "https://www.youtube.com/watch?v=lPdmYMHrWKg",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-df6153da",
            "type": "article",
            "title": "Resource for AI Safety and Ethics",
            "summary": "",
            "url": "https://library.iaseai.org/",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-ce4372d9",
            "type": "article",
            "title": "In this era of AI, it is important to remember what it is like to be a child",
            "summary": "",
            "url": "https://mathstodon.xyz/@tao/117244104044239500",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-e68be5e8",
            "type": "article",
            "title": "Anthropic Researcher Quits over 'Out-of-Control' AI Fears",
            "summary": "",
            "url": "https://www.wsj.com/tech/ai/anthropic-researcher-quits-over-out-of-control-ai-fears-707b7628",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-6798d04e",
            "type": "article",
            "title": "Nvidia Personal-AI-Router",
            "summary": "",
            "url": "https://github.com/NVIDIA/Personal-AI-Router",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-1d5ebb0a",
            "type": "article",
            "title": "Reducing cost and improving performance with Claude Platform",
            "summary": "",
            "url": "https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-d8e70a90",
            "type": "article",
            "title": "Ask HN: How do you kill time while LLMs work for you?",
            "summary": "",
            "url": "https://news.ycombinator.com/item?id=49640293",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-27114c3f",
            "type": "article",
            "title": "Kelivo: A Flutter LLM Chat Client. Support Mobile and Desktop",
            "summary": "",
            "url": "https://github.com/Chevey339/kelivo",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-42b3371d",
            "type": "article",
            "title": "AI Coding Assistants Almost Never Check Supply-Chain Trust Signals",
            "summary": "",
            "url": "https://arxiv.org/abs/2609.07754",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-30db0b2b",
            "type": "article",
            "title": "RL trained a 4B VLM to play GeoGuesser",
            "summary": "",
            "url": "https://huggingface.co/spaces/HuggingEnvs/geoguesser-article",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-7c0156ff",
            "type": "article",
            "title": "Can LLM Agents Infer World Models? Evidence from Agentic Automata Learning",
            "summary": "",
            "url": "https://arxiv.org/abs/2606.16576",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-1852353d",
            "type": "article",
            "title": "Training a 3.8B LLM to 0.384 CORE for $998",
            "summary": "",
            "url": "https://hugovergnes.github.io/little-lm-3-8b/",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 58
        },
        {
            "id": "article-e166c7b5",
            "type": "article",
            "title": "Show HN: Ambient Context v1.0.0 (Missing context for AI assistants)",
            "summary": "",
            "url": "https://github.com/dragthelake/ambient-context/releases",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-ca2af7b3",
            "type": "article",
            "title": "Show HN: An open-weight LLM whose answer/stop decision can be flipped internally",
            "summary": "",
            "url": "https://github.com/theonlypal/PCCG-Qwen3-4B-continuation-control",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-a6645e50",
            "type": "article",
            "title": "Schools Get 5M Complaints yet Almost No Teachers Trained to Handle Them",
            "summary": "",
            "url": "https://replyresearch.com/schools-get-5m-complaints-yet-almost-no-teachers-trained-to-handle-them/",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-9745d303",
            "type": "article",
            "title": "Anthropic commerce agent: open-source blueprint for shopping and merchant agents",
            "summary": "",
            "url": "https://github.com/anthropics/commerce-agents",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-8794adbd",
            "type": "article",
            "title": "Show HN: ColliePWA a mobile interface for AI agents in herdr/tmux/zellij",
            "summary": "",
            "url": "https://colliepwa.dev/",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-a5bc3177",
            "type": "article",
            "title": "Structure Around Intelligence \u2013 AI Native",
            "summary": "",
            "url": "https://news.ycombinator.com/item?id=49576482",
            "source": "hackernews",
            "date": "2026-09-05",
            "trendingScore": 50
        },
        {
            "id": "article-3f3dede7",
            "type": "article",
            "title": "Show HN: TERMy \u2013 A fast terminal assistant that does not use LLMs",
            "summary": "",
            "url": "https://github.com/gioblu/NPC-Forge/blob/main/docs/development.md",
            "source": "hackernews",
            "date": "2026-09-04",
            "trendingScore": 72
        },
        {
            "id": "article-089296cc",
            "type": "article",
            "title": "Ask HN: Can we classify AI as: subhuman, quasihuman, human, superhumam?",
            "summary": "",
            "url": "https://news.ycombinator.com/item?id=49552230",
            "source": "hackernews",
            "date": "2026-09-03",
            "trendingScore": 50
        },
        {
            "id": "article-397dd74b",
            "type": "article",
            "title": "Neural networks reveal how experience shapes learning in both brains n machines",
            "summary": "",
            "url": "https://medicalxpress.com/news/2026-09-neural-networks-reveal-brains-machines.html",
            "source": "hackernews",
            "date": "2026-09-03",
            "trendingScore": 50
        },
        {
            "id": "article-58fc1579",
            "type": "article",
            "title": "Show HN; AGI brain project. Who wants to join in and help me?",
            "summary": "",
            "url": "https://news.ycombinator.com/item?id=49623800",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-6adb162a",
            "type": "article",
            "title": "Outrageously Small Neural Networks: Emergent Basic Reasoning at 6,616 tok/SEC [pdf]",
            "summary": "",
            "url": "https://huggingface.co/gdiamos/amx-reasoning-v1-instruct/blob/main/paper.pdf",
            "source": "hackernews",
            "date": "2026-09-08",
            "trendingScore": 50
        },
        {
            "id": "article-5d1394e9",
            "type": "article",
            "title": "Physics-Informed Neural Networks \u2013 Making Your Neural Networks Even Smarter",
            "summary": "",
            "url": "https://abelabraham3.substack.com/p/the-rabbit-hole-2-making-your-neural",
            "source": "hackernews",
            "date": "2026-09-04",
            "trendingScore": 50
        },
        {
            "id": "article-c30b4e3f",
            "type": "article",
            "title": "You're also a neural network \u2013 train on the good stuff",
            "summary": "",
            "url": "https://magnusludviksson.substack.com/p/youre-also-a-neural-network-train",
            "source": "hackernews",
            "date": "2026-09-03",
            "trendingScore": 50
        },
        {
            "id": "article-25e60bad",
            "type": "article",
            "title": "Benchmarking Claude Code, Codex and Pi on SWE-Bench Pro: Same Accuracy, 2x Cost",
            "summary": "",
            "url": "https://aistack.imec-int.com/blog/harness-cost",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-baf4c2eb",
            "type": "article",
            "title": "Hackers are stealing Claude tokens from subscribers",
            "summary": "",
            "url": "https://techcrunch.com/2026/09/08/hackers-are-stealing-claude-tokens-from-subscribers/",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-163b76cf",
            "type": "article",
            "title": "Show HN: Claude Project Downloader",
            "summary": "",
            "url": "https://chromewebstore.google.com/detail/claude-project-downloader/ghbdnpphggjonapjkehkmhpnmhcngbbg",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-e5539435",
            "type": "article",
            "title": "Anthropic discloses fourth AI hacking incident missed in earlier review",
            "summary": "",
            "url": "https://www.reuters.com/legal/litigation/anthropic-reports-fourth-cybersecurity-incident-with-early-version-claude-2026-09-09/",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 51
        },
        {
            "id": "article-980f8886",
            "type": "article",
            "title": "GPT-6 Astra is better at making money, more ethical than Claude Fable 5.1",
            "summary": "",
            "url": "https://twitter.com/andonlabs/status/2097377692966633952",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-c48809a4",
            "type": "article",
            "title": "Mins after I cancelled my Claude subscription I was rejected from Anthropic",
            "summary": "",
            "url": "https://twitter.com/par_dot_dev/status/2097777825369530450",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-53d162f6",
            "type": "article",
            "title": "Ask HN: Codex conversations failing to converge across devices?",
            "summary": "",
            "url": "https://news.ycombinator.com/item?id=49632749",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-308295b7",
            "type": "article",
            "title": "LittleSwitch, my personnal router for Claude/Codex (Desktop/CLI/Cowork) on macOS",
            "summary": "",
            "url": "https://little-switch.alfredlabs.io",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-6ff1582c",
            "type": "article",
            "title": "Contextify for Windows \u2013 Live Index of Your Claude Code/Codex History",
            "summary": "",
            "url": "https://contextify.sh/windows/",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-9a4778fa",
            "type": "article",
            "title": "Responding to Claude's feedback prompts is opting-in for data sharing",
            "summary": "",
            "url": "https://privacy.claude.com/en/articles/10023580-is-my-data-used-for-model-training",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-f2fa6d32",
            "type": "article",
            "title": "Show HN: Self-hosted company OS, Claude Code and Codex agents in departments",
            "summary": "",
            "url": "https://github.com/OtoDock/oto-dock",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 54
        },
        {
            "id": "article-79afdf13",
            "type": "article",
            "title": "Show HN: Hordev \u2013 Claude Code skills that build instead of asking questions",
            "summary": "",
            "url": "https://github.com/heffrey/hordev",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-c11252a1",
            "type": "article",
            "title": "Use 1Password to sign in to websites with Claude",
            "summary": "",
            "url": "https://support.1password.com/1password-claude/",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-c05bfcae",
            "type": "article",
            "title": "Sergey Brin is cooking up Google's AI destiny",
            "summary": "",
            "url": "https://www.businessinsider.com/sergey-brin-google-gemini-ai-microkitchen-2026-9",
            "source": "hackernews",
            "date": "2026-09-10",
            "trendingScore": 50
        },
        {
            "id": "article-fc7dd102",
            "type": "article",
            "title": "Show HN: Castforge \u2013 run Claude Code, Codex and Gemini as one dev team",
            "summary": "",
            "url": "https://castforge.ai/",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-bea77d6c",
            "type": "article",
            "title": "Is Google planning for Gemini 4 rather than 3.5 pro?",
            "summary": "",
            "url": "https://news.ycombinator.com/item?id=49624779",
            "source": "hackernews",
            "date": "2026-09-09",
            "trendingScore": 50
        },
        {
            "id": "article-67069f08",
            "type": "article",
            "title": "Show HN: SiteTell: finds the areas of your site that read as AI-generic",
            "summary": "",
            "url": "https://www.getsitetell.com/",
            "source": "hackernews",
            "date": "2026-09-08",
            "trendingScore": 50
        },
        {
            "id": "article-5479a8e2",
            "type": "article",
            "title": "How to export Gemini chats to pdf [video]",
            "summary": "",
            "url": "https://www.youtube.com/watch?v=zFdp8DUfDco",
            "source": "hackernews",
            "date": "2026-09-08",
            "trendingScore": 50
        },
        {
            "id": "article-df6142cf",
            "type": "article",
            "title": "Agentic Video Understanding with Gemini",
            "summary": "",
            "url": "https://twitter.com/GoogleAIStudio/status/2094848490203410564",
            "source": "hackernews",
            "date": "2026-09-08",
            "trendingScore": 50
        },
        {
            "id": "article-428e3b6f",
            "type": "article",
            "title": "Hikers stranded on Mount Shasta after following a plan by Gemini AI",
            "summary": "",
            "url": "https://www.latimes.com/california/story/2026-09-03/hikers-following-google-gemini-ai-route-become-stranded-on-mt-shasta",
            "source": "hackernews",
            "date": "2026-09-07",
            "trendingScore": 50
        },
        {
            "id": "article-8455d416",
            "type": "article",
            "title": "Google's Gemini Spark can now manage your Google Photos library",
            "summary": "",
            "url": "https://techcrunch.com/2026/09/04/googles-gemini-spark-can-now-manage-your-google-photos-library/",
            "source": "hackernews",
            "date": "2026-09-07",
            "trendingScore": 50
        },
        {
            "id": "article-4fe7a00c",
            "type": "article",
            "title": "I track LLM API prices daily, found a 33x cost gap in the same model",
            "summary": "",
            "url": "https://news.ycombinator.com/item?id=49586810",
            "source": "hackernews",
            "date": "2026-09-06",
            "trendingScore": 50
        },
        {
            "id": "article-a67a9a88",
            "type": "article",
            "title": "Show HN: I stopped using an LLM gateway and put rate-limits/fallback in-process",
            "summary": "",
            "url": "https://github.com/LakBud/vernLLM",
            "source": "hackernews",
            "date": "2026-09-06",
            "trendingScore": 50
        },
        {
            "id": "article-f774e48a",
            "type": "article",
            "title": "Hikers rescued after using Google Gemini for planning",
            "summary": "",
            "url": "https://techcrunch.com/2026/09/05/hikers-rescued-after-using-google-gemini-for-planning/",
            "source": "hackernews",
            "date": "2026-09-05",
            "trendingScore": 50
        },
        {
            "id": "article-301ab5a9",
            "type": "article",
            "title": "Gemini 4 Leak. Google is back",
            "summary": "",
            "url": "https://twitter.com/ravikiran_dev7/status/2095869859666268606",
            "source": "hackernews",
            "date": "2026-09-05",
            "trendingScore": 50
        },
        {
            "id": "article-3da79424",
            "type": "article",
            "title": "Gemini vs. Claude: Which AI Model created a better perfume?",
            "summary": "",
            "url": "https://www.youtube.com/watch?v=YfFk050AjPw",
            "source": "hackernews",
            "date": "2026-09-04",
            "trendingScore": 50
        },
        {
            "id": "article-1fb2da1f",
            "type": "article",
            "title": "Show HN: Tesoro.help \u2013 rogue AI helpdesk for my kid's high school",
            "summary": "",
            "url": "https://tesoro.help/",
            "source": "hackernews",
            "date": "2026-09-04",
            "trendingScore": 50
        },
        {
            "id": "article-17360fb7",
            "type": "article",
            "title": "Show HN: Jigsaw Haiku",
            "summary": "",
            "url": "https://jigsawhaiku.com/",
            "source": "hackernews",
            "date": "2026-09-04",
            "trendingScore": 61
        },
        {
            "id": "article-35d90255",
            "type": "article",
            "title": "Show HN: Cloud Bill Billionaire",
            "summary": "",
            "url": "https://loganbesecker.com/Apps/Cloud-Bill-Billionaire.html",
            "source": "hackernews",
            "date": "2026-09-04",
            "trendingScore": 50
        },
        {
            "id": "topic-large-language-models",
            "type": "topic",
            "title": "Large Language Models",
            "summary": "Foundation models trained on massive text corpora that can generate and understand natural language.",
            "connectionCount": 37
        },
        {
            "id": "topic-ai-reasoning",
            "type": "topic",
            "title": "AI Reasoning",
            "summary": "Methods to improve logical reasoning, mathematical problem-solving, and multi-step thinking in AI systems.",
            "connectionCount": 5
        },
        {
            "id": "topic-ai-safety",
            "type": "topic",
            "title": "AI Safety",
            "summary": "Research focused on making AI systems safe, aligned with human values, and beneficial.",
            "connectionCount": 5
        },
        {
            "id": "topic-nlp",
            "type": "topic",
            "title": "NLP",
            "summary": "Natural Language Processing: AI techniques for understanding and generating human language.",
            "connectionCount": 28
        },
        {
            "id": "topic-ai-agents",
            "type": "topic",
            "title": "AI Agents",
            "summary": "Autonomous AI systems that can plan, use tools, and take actions to accomplish goals.",
            "connectionCount": 21
        },
        {
            "id": "topic-reinforcement-learning",
            "type": "topic",
            "title": "Reinforcement Learning",
            "summary": "Training AI through rewards and penalties to learn optimal behaviors.",
            "connectionCount": 26
        },
        {
            "id": "topic-model-efficiency",
            "type": "topic",
            "title": "Model Efficiency",
            "summary": "Techniques to reduce computational costs and improve inference speed of AI models.",
            "connectionCount": 6
        },
        {
            "id": "topic-computer-vision",
            "type": "topic",
            "title": "Computer Vision",
            "summary": "AI systems for understanding and processing visual information from images and video.",
            "connectionCount": 6
        },
        {
            "id": "topic-prompt-engineering",
            "type": "topic",
            "title": "Prompt Engineering",
            "summary": "Methods for crafting effective prompts to guide AI model behavior and outputs.",
            "connectionCount": 4
        },
        {
            "id": "topic-rag",
            "type": "topic",
            "title": "RAG",
            "summary": "Retrieval-Augmented Generation: combining LLMs with external knowledge retrieval for more accurate responses.",
            "connectionCount": 8
        },
        {
            "id": "topic-fine-tuning",
            "type": "topic",
            "title": "Fine-tuning",
            "summary": "Adapting pre-trained models to specific tasks or domains.",
            "connectionCount": 4
        },
        {
            "id": "topic-diffusion-models",
            "type": "topic",
            "title": "Diffusion Models",
            "summary": "Generative models that create content by iteratively denoising random noise into structured outputs.",
            "connectionCount": 1
        },
        {
            "id": "topic-multimodal-ai",
            "type": "topic",
            "title": "Multimodal AI",
            "summary": "Systems that process and understand multiple types of input including text, images, audio, and video.",
            "connectionCount": 3
        },
        {
            "id": "org-meta",
            "type": "organization",
            "title": "Meta",
            "summary": "Meta - AI research and development.",
            "connectionCount": 3
        },
        {
            "id": "org-google",
            "type": "organization",
            "title": "Google",
            "summary": "Google - AI research and development.",
            "connectionCount": 7
        },
        {
            "id": "org-ibm",
            "type": "organization",
            "title": "IBM",
            "summary": "IBM - AI research and development.",
            "connectionCount": 1
        },
        {
            "id": "org-openai",
            "type": "organization",
            "title": "OpenAI",
            "summary": "OpenAI - AI research and development.",
            "connectionCount": 1
        },
        {
            "id": "org-nvidia",
            "type": "organization",
            "title": "NVIDIA",
            "summary": "NVIDIA - AI research and development.",
            "connectionCount": 2
        },
        {
            "id": "org-anthropic",
            "type": "organization",
            "title": "Anthropic",
            "summary": "Anthropic - AI research and development.",
            "connectionCount": 4
        },
        {
            "id": "model-gemini",
            "type": "model",
            "title": "Gemini",
            "summary": "Gemini AI model.",
            "connectionCount": 11
        },
        {
            "id": "model-claude",
            "type": "model",
            "title": "Claude",
            "summary": "Claude AI model.",
            "connectionCount": 15
        }
    ],
    "edges": [
        {
            "source": "article-8b25c51e",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-8b25c51e",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-8b25c51e",
            "target": "topic-ai-safety",
            "relationship": "COVERS"
        },
        {
            "source": "article-8b25c51e",
            "target": "org-meta",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-e23485c0",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-e23485c0",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-cb56f2d0",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-cb56f2d0",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-7768e631",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-7768e631",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-40be5fe8",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-40be5fe8",
            "target": "topic-model-efficiency",
            "relationship": "COVERS"
        },
        {
            "source": "article-40be5fe8",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-f861f5c8",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-f861f5c8",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-f861f5c8",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-5829b12a",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-ec956445",
            "target": "topic-model-efficiency",
            "relationship": "COVERS"
        },
        {
            "source": "article-99b54e34",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-99b54e34",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-390753c0",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-390753c0",
            "target": "topic-prompt-engineering",
            "relationship": "COVERS"
        },
        {
            "source": "article-34398729",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-34398729",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-34398729",
            "target": "topic-model-efficiency",
            "relationship": "COVERS"
        },
        {
            "source": "article-34398729",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-63ad0e1f",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-63ad0e1f",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-63ad0e1f",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-eda0e237",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-eda0e237",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-eda0e237",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-88e4e50e",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-88e4e50e",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-9d857600",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-9d857600",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-9d857600",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-9d857600",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-9d857600",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-8ad8a572",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-8ad8a572",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-8ad8a572",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-386b820e",
            "target": "topic-model-efficiency",
            "relationship": "COVERS"
        },
        {
            "source": "article-320cc556",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-320cc556",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-320cc556",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-320cc556",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-269f1f39",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-269f1f39",
            "target": "topic-prompt-engineering",
            "relationship": "COVERS"
        },
        {
            "source": "article-de919604",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-de919604",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-5b8248bb",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-5b8248bb",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-5b8248bb",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-1e913121",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-1e913121",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-1e913121",
            "target": "topic-fine-tuning",
            "relationship": "COVERS"
        },
        {
            "source": "article-a843ad46",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-a843ad46",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-a843ad46",
            "target": "org-meta",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-7ad1ccf6",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-7ad1ccf6",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-6771e1ef",
            "target": "topic-diffusion-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-5b88a2c1",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-b9e6ee3d",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-6156280c",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-6156280c",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-6156280c",
            "target": "topic-fine-tuning",
            "relationship": "COVERS"
        },
        {
            "source": "article-6ce925d4",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-d9add56a",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-70d4862f",
            "target": "topic-model-efficiency",
            "relationship": "COVERS"
        },
        {
            "source": "article-70d4862f",
            "target": "topic-fine-tuning",
            "relationship": "COVERS"
        },
        {
            "source": "article-7da30aa3",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-7da30aa3",
            "target": "topic-prompt-engineering",
            "relationship": "COVERS"
        },
        {
            "source": "article-7da30aa3",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-67257591",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-67257591",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-67257591",
            "target": "org-meta",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-f992524a",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-f992524a",
            "target": "topic-ai-safety",
            "relationship": "COVERS"
        },
        {
            "source": "article-f992524a",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-f3ce0512",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-334d3757",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-334d3757",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-d8e070b6",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-d8e070b6",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-d8e070b6",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-4cd9b4ff",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-4cd9b4ff",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-4cd9b4ff",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-98a5601a",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-6702cf8a",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-6702cf8a",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-6702cf8a",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-52c2df42",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-52c2df42",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-52c2df42",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-aa2fe68f",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-21a4ea1d",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-3a013d6c",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-3a013d6c",
            "target": "topic-model-efficiency",
            "relationship": "COVERS"
        },
        {
            "source": "article-c4703e85",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-c4703e85",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-c4703e85",
            "target": "topic-fine-tuning",
            "relationship": "COVERS"
        },
        {
            "source": "article-2e8f68d6",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-2e8f68d6",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-2e8f68d6",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-8fe51aaf",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-8fe51aaf",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-8fe51aaf",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-ac5f43f0",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-434c41dd",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-434c41dd",
            "target": "topic-multimodal-ai",
            "relationship": "COVERS"
        },
        {
            "source": "article-27cd6838",
            "target": "topic-ai-safety",
            "relationship": "COVERS"
        },
        {
            "source": "article-27cd6838",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-6b48ca84",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-ec5828eb",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-8c02b877",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-8c02b877",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-8c02b877",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-8c02b877",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-8c02b877",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-43d2948c",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-32522e85",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-0e4c7024",
            "target": "org-ibm",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-c847825e",
            "target": "topic-ai-safety",
            "relationship": "COVERS"
        },
        {
            "source": "article-aad76210",
            "target": "topic-multimodal-ai",
            "relationship": "COVERS"
        },
        {
            "source": "article-388663d3",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-388663d3",
            "target": "org-openai",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-f4a18652",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-f4a18652",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-c5405304",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-c5405304",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-16db96c9",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-b0b4d640",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-b0b4d640",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-5bc17ec6",
            "target": "org-nvidia",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-0ab30f68",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-448d938b",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-df6153da",
            "target": "topic-ai-safety",
            "relationship": "COVERS"
        },
        {
            "source": "article-e68be5e8",
            "target": "org-anthropic",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-6798d04e",
            "target": "org-nvidia",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-1d5ebb0a",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-d8e70a90",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-27114c3f",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-30db0b2b",
            "target": "topic-multimodal-ai",
            "relationship": "COVERS"
        },
        {
            "source": "article-30db0b2b",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-7c0156ff",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-7c0156ff",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-7c0156ff",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-1852353d",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-e166c7b5",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-ca2af7b3",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-9745d303",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-9745d303",
            "target": "org-anthropic",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-8794adbd",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-3f3dede7",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-6adb162a",
            "target": "topic-ai-reasoning",
            "relationship": "COVERS"
        },
        {
            "source": "article-6adb162a",
            "target": "topic-rag",
            "relationship": "COVERS"
        },
        {
            "source": "article-25e60bad",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-baf4c2eb",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-163b76cf",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-e5539435",
            "target": "topic-reinforcement-learning",
            "relationship": "COVERS"
        },
        {
            "source": "article-e5539435",
            "target": "org-anthropic",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-980f8886",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-c48809a4",
            "target": "org-anthropic",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-c48809a4",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-308295b7",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-6ff1582c",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-6ff1582c",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-9a4778fa",
            "target": "topic-prompt-engineering",
            "relationship": "COVERS"
        },
        {
            "source": "article-9a4778fa",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-f2fa6d32",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-f2fa6d32",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-79afdf13",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-c11252a1",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-c05bfcae",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-fc7dd102",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-fc7dd102",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-bea77d6c",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-bea77d6c",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-67069f08",
            "target": "topic-nlp",
            "relationship": "COVERS"
        },
        {
            "source": "article-5479a8e2",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-5479a8e2",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-df6142cf",
            "target": "topic-ai-agents",
            "relationship": "COVERS"
        },
        {
            "source": "article-df6142cf",
            "target": "topic-computer-vision",
            "relationship": "COVERS"
        },
        {
            "source": "article-df6142cf",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-428e3b6f",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-8455d416",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-8455d416",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-4fe7a00c",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-a67a9a88",
            "target": "topic-large-language-models",
            "relationship": "COVERS"
        },
        {
            "source": "article-f774e48a",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-f774e48a",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-301ab5a9",
            "target": "org-google",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-301ab5a9",
            "target": "model-gemini",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-3da79424",
            "target": "model-claude",
            "relationship": "MENTIONS"
        },
        {
            "source": "article-3da79424",
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
