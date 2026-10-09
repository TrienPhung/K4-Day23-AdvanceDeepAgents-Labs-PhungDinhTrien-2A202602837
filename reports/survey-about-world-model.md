# World Models in AI: From Predictive Representations to Decision-Useful Simulators

## TL;DR
- “World model” covers both an agent’s internal account of how the world works and a model that predicts future states; the latter becomes especially useful when predictions are conditioned on candidate actions. [1][2]
- The field spans compact latent dynamics for planning and control, through to generative video environments; these approaches optimize different trade-offs between decision relevance and perceptual detail. [3][4][5]
- Evaluation is moving beyond visual plausibility: embodied benchmarks test whether imagined actions can be executed and achieve goals, while physics benchmarks probe consistency with physical concepts. [6][7][8]
- Recent work identifies a persistent gap between scaling or visual quality and reliable physical generalization, making uncertainty, long-horizon consistency, and decision-centered testing central open problems. [9][10][11]

## Background
A world model is not a single architecture or settled technical definition. One useful distinction is between (i) an internal representation intended to capture mechanisms of the external world and (ii) a predictive model of future states or dynamics, used to simulate what may happen and guide decisions. The first emphasizes what an agent represents; the second emphasizes what it can predict. The two overlap when a model predicts consequences of actions in a way that supports planning. [1]

A key foundation for the modern predictive strand is the 2018 “World Models” work: generative neural networks learned compressed spatial and temporal representations of an environment, whose features informed a compact policy. The paper also reported training an agent inside a generated “dream” and transferring the policy to the real environment. [2] PlaNet made the planning connection explicit by learning dynamics from images and selecting actions through online planning in latent space; its model combined deterministic and stochastic transition components. [3] These works established a durable design principle: an agent need not plan over full-resolution observations if a compact predictive state retains the information needed for action.

The term has broadened as video generation and multimodal systems have become more capable. A recent survey describes renewed interest following prominent video-generation and multimodal systems, but stresses that whether such systems constitute comprehensive world models remains an open question. [1] Thus, “world model” is best treated as a family of approaches, distinguished by what they predict, how they are conditioned, and whether their predictions can improve decisions—not as a synonym for realistic generated video.

## Theme 1: Predictive state, latent planning, and imagined experience

In model-based reinforcement learning, a learned model can provide simulated experience for policy improvement. DreamerV3 is a contemporary example of the latent-imagination route: it improves behavior by imagining future scenarios in a learned environment model. The authors report one configuration across more than 150 diverse tasks, including collecting Minecraft diamonds from scratch without human data or curricula. [12] The relevance is not simply that it generates futures, but that its predictions are embedded in a decision loop.

This differs from a video generator whose primary output is a plausible future clip. In predictive control, a compact state and reward- or task-relevant predictions may matter more than reconstructing every pixel. VFMF makes this contrast directly: it forecasts vision-foundation-model features rather than future RGB frames, arguing that feature prediction can be more computationally efficient than stochastic pixel-video generation and can feed decisions without an additional conversion from pixels to task signals. [13] The contrast is not that one representation is universally superior: latent or feature-space prediction prioritizes compactness and downstream use, whereas pixel generation can retain detailed visual appearance that may be useful for simulation and human inspection.

## Theme 2: Generative, interactive, and action-conditioned worlds

A newer line turns learned video dynamics into environments that users or agents can control. Genie was trained on unlabelled Internet videos and combines a spatiotemporal tokenizer, autoregressive dynamics model, and latent-action model; it generates action-controllable virtual worlds that can be interacted with frame by frame. [4] Its latent actions are useful where videos lack explicit action labels, but the note-supported evidence describes controllability in the generated environment rather than establishing that its actions transfer directly to a physical robot.

Other approaches adapt pretrained video generators. Dynamic World Simulation adds action-conditioned modules and a motion-reinforced loss to video-generation models, targeting more controllable and dynamically consistent world simulation. [5] PAN takes a hybrid route, combining autoregressive latent dynamics with a video-diffusion decoder to pursue detailed, coherent, long-term interactive simulation. [14] These examples illustrate a recurring architectural compromise: latent dynamics can organize temporal and action structure, while a video decoder supplies visual detail. The practical question is whether the generated detail is consistent with the underlying state transition, rather than merely convincing frame by frame.

A different compromise separates action dynamics from appearance. VideoWorld 2 proposes a dynamic-enhanced latent dynamics model learned from raw real-world video, with a pretrained video diffusion model handling appearance; its latent codes are intended to capture transferable action dynamics. [15] Collectively, these approaches broaden the scope from “predict the next state for a controller” to “construct an interactive environment,” but they also make consistency over long rollouts and meaningful action control more demanding evaluation targets.

## Theme 3: Applications and what benchmarks actually measure

World models support several distinct goals: improving control through imagined experience, evaluating or planning actions against simulated futures, and producing synthetic environments or data. These aims should not be collapsed into one quality score. In robotics, RoboWM-Bench evaluates whether behaviors predicted in videos can be translated into robot action sequences, using reconstructed simulation scenes and execution-based validation. [7]

WorldArena likewise distinguishes perception from functional utility. Its evaluation covers video quality across visual, motion, consistency, physics, 3D, and controllability dimensions, as well as use as a data engine, policy evaluator, or action planner; planning is tested by executing predicted sequences in RoboTwin and measuring task success. It reports a perception–functionality gap across 14 representative models. [6] These benchmarks therefore ask a more demanding question than “does the future look plausible?”: can a model’s predictions support an action that works?

Physical-video benchmarks provide a complementary test. PhyGenBench organizes 160 prompts around 27 physical laws in four domains and reports that evaluated video models struggle to generate physically compliant results; scaling or prompt engineering alone is insufficient for some dynamic scenarios. [8] WorldBench isolates concepts such as motion, object permanence, support, scale and perspective, and tests physical quantities including gravity, viscosity, and friction using real and simulated videos. [16] Meanwhile, EWMBench scores embodied video-generation systems along scene, motion, and semantic quality. [17] The different emphases matter: appearance and semantic consistency, explicit physical-law adherence, and action executability measure related but non-equivalent capabilities.

## Theme 4: Scaling, reliability, and the limits of simulation

Recent findings caution against treating more data or larger models as a general solution. A scaling-law study reports power-law relationships between loss and optimal model size for world modeling and imitation learning, but says coefficients depend strongly on tokenizer, task, and architecture. [18] In a separate physical-law evaluation, video models show in-distribution and measurable combinatorial generalization but fail on out-of-distribution scenarios, exhibiting case-based rather than general physical-rule behavior; the authors conclude scaling alone is insufficient to uncover fundamental physical laws. [10] Scaling can therefore improve measured performance without guaranteeing robust causal or physical understanding.

Reliability is especially consequential when a model is used for long-horizon control. A survey of embodied world models identifies long-horizon consistency and error accumulation, real-time compute, scarce unified datasets, and the need to prioritize physical consistency over pixel fidelity as central challenges. [9] Work on offline robotic model-based RL likewise identifies compounding rollout errors and distribution shift, particularly with noisy, biased, partially observed real-robot datasets, and proposes an uncertainty-aware model as one response. [11] A safety review further calls for controllable generation, quantitative uncertainty evaluation, guardrails, and benchmarks covering physical and social plausibility. [19]

## Trends and open problems

Across the recent literature, “world modeling” is expanding in two directions: decision-coupled models designed to improve planning and control, and general-purpose interactive simulators built around video generation. The former foregrounds action-conditioned dynamics and task-relevant prediction; the latter foregrounds broad coverage and rich visual futures. Hybrid designs seek both, but the sources do not establish that visual realism automatically yields reliable dynamics or usable controls. [1][5][14]

Three open problems follow. First, evaluation needs to link predictions to downstream decisions and execution, while separately measuring visual, physical, and semantic quality; benchmark results already show that these axes can diverge. [6][7][17] Second, generalization beyond familiar visual situations remains unresolved: reported out-of-distribution physical reasoning failures challenge claims that scaling video models alone will produce general physical laws. [10] Third, practical agents need calibrated uncertainty, safe control, and efficient predictions over long horizons, particularly when trained on imperfect real-world data. [9][19][11] The field’s central test is therefore not whether a model can imagine a future, but whether its imagined futures remain physically and decision-relevant when actions, environments, or time horizons change.

## References
[1] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[2] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-03-27)
[3] Learning Latent Dynamics for Planning from Pixels. arxiv. https://arxiv.org/abs/1811.04551 (2018-11-12)
[4] Genie: Generative Interactive Environments. arxiv. https://arxiv.org/abs/2402.15391 (2024-02-23)
[5] Pre-Trained Video Generative Models as World Simulators. hf-search. https://huggingface.co/papers/2502.07825 (2025-02-10)
[6] WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models. web. https://arxiv.org/html/2602.08971v1 (n.d.)
[7] RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation. web. https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/papers/Jiang_RoboWM-Bench_A_Benchmark_for_Evaluating_World_Models_in_Robotic_Manipulation_CVPRW_2026_paper.pdf (n.d.)
[8] Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation. web. https://proceedings.mlr.press/v267/meng25c.html (2025-10-06)
[9] A Comprehensive Survey on World Models for Embodied AI. web. https://arxiv.org/html/2510.16732v1 (2025-10-19)
[10] How Far Is Video Generation from World Model: A Physical Law Perspective. web. https://proceedings.mlr.press/v267/kang25g.html (2025-10-06)
[11] Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots. arxiv. https://arxiv.org/abs/2504.16680 (2025-04-23)
[12] Mastering diverse control tasks through world models. web. https://www.nature.com/articles/s41586-025-08744-2 (2025-04-02)
[13] VFMF: World Modeling by Forecasting Vision Foundation Model Features. arxiv. https://arxiv.org/abs/2512.11225 (2025-12-12)
[14] PAN: A World Model for General, Interactable, and Long-Horizon World Simulation. hf-search. https://huggingface.co/papers/2511.09057 (2025-11-12)
[15] VideoWorld 2: Learning Transferable Knowledge from Real-world Videos. arxiv. https://arxiv.org/abs/2602.10102 (2026-02-10)
[16] WorldBench: Benchmarking Physical Understanding of World Models by Isolating Physics Concepts. web. https://world-bench.github.io/ (n.d.)
[17] EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models. hf-search. https://huggingface.co/papers/2505.09694 (2025-05-14)
[18] Scaling Laws for Pre-training Agents and World Models. web. https://proceedings.mlr.press/v267/pearce25a.html (2025-10-06)
[19] The Safety Challenge of World Models for Embodied AI Agents: A Review. web. https://arxiv.science/abs/2510.05865 (2025-10-07)
