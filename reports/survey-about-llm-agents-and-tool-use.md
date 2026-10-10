# LLM Agents and Tool Use: Methods, Evaluation, and Safety

## TL;DR
- Tool-using agents extend text generation into action: they decide what to do, invoke tools or environments, and incorporate the resulting feedback into subsequent reasoning or plans. [1][2]
- The field has moved from prompting models to act in a loop toward building tool competence through API-oriented training and coherent tool-use data. [3][4][5]
- Evaluation is becoming more granular: benchmarks distinguish choosing whether to call a tool, executing API interactions, completing interactive tasks, and judging full trajectories. [6][7][8][9]
- Tool access creates security risks because external content can manipulate actions; security benchmarks therefore test adversarial behavior alongside task utility. [10][11][12]

## Background
An LLM agent is not simply a model that mentions a tool. It is a system that interacts with an external tool or environment, receives observations, and uses them to select further actions. One influential framing interleaves reasoning traces with task-specific actions and external feedback, allowing plans to be created or revised as the interaction proceeds. [1][2] This matters because language models have limits in functions such as arithmetic and factual lookup, while external APIs and environments can provide operations or information unavailable in the model’s own text generation. [3]

Agent design consequently spans more than the call syntax. A survey taxonomy identifies policy, evaluator, and dynamic-model roles and distinguishes direct state-to-action mapping from planning an action sequence. [2] Another survey organizes construction around profile, memory, planning, and action execution, describing a feedback loop in which execution outcomes inform later planning and memory. [13] These perspectives help separate agent architecture from the narrower question of whether a model can produce a syntactically valid function call.

## From action loops to persistent agent architectures
ReAct offers a compact interaction pattern: the model alternates verbal reasoning and actions, then observes external results before continuing. It was evaluated across question answering, fact verification, text-based games, and web navigation; the paper reports gains over compared imitation- or reinforcement-learning methods on ALFWorld and WebShop. [1] The broader architectural lesson is that tool use is often iterative rather than a single request-response: observations can change what the system should do next. [1][2]

Other designs add persistent experience and explicit planning. Generative Agents store natural-language records of experiences, synthesize higher-level reflections, and retrieve memories to guide behavior; in an interactive sandbox, the work instantiated 25 agents. [14] This is complementary to ReAct’s immediate action-feedback cycle: one emphasizes grounding the next step in current external feedback, while the other illustrates how accumulated experience can influence later plans. [1][14]

The distinction between acting and planning also matters operationally. In the survey framing, an actor maps a state to an action, whereas a planner proposes an action sequence; evaluator roles can assess actions or resulting states. [2] For tool-rich tasks, a plan may need revision after a tool returns unexpected or incomplete information, which makes feedback part of the architecture rather than merely a final scoring signal. [1][2]

## Building tool competence: selection, arguments, and training
A useful decomposition of tool use is deciding whether to call a tool, selecting which tool to use, generating its arguments, and incorporating the result. Toolformer addresses when and how APIs should be called, what arguments to pass, and how to use results in future token prediction. [3] Gorilla likewise highlights argument accuracy and hallucinated API usage as challenges in API calling. [15] These are related but distinct failure points: a model may choose an unsuitable API, form invalid parameters, or fail to use the returned information effectively. [3][15]

Scaling tool use has prompted work on data and training pipelines. ToolLLM presents a framework covering data construction, model training, and evaluation for a large collection of real-world APIs, while noting that tool-use capability has been neglected in some instruction tuning. [4] Tool Learning in the Wild identifies manual documentation parsing and in-context examples as approaches that do not scale easily to large toolsets, and points to integration challenges between free-form generation and tool-calling actions. [16]

Data quality is also a concern. ToolFlow focuses on supervised fine-tuning with synthesized dialogue and argues that randomly sampled tools can yield irrelevant combinations, while incoherent turns leave a gap from realistic interactions. [5] This complements API coverage efforts: broad tool inventories do not by themselves guarantee that training examples teach a coherent multi-turn interaction. [4][5] In specialized settings, function calling can expose structured operations over changing or private databases; one database-query design unifies search queries, filters, their combinations, and result aggregation. [17]

## Evaluation: from call decisions to task outcomes
No single benchmark dimension captures all tool-use ability. When2Call isolates the decision of whether to call a tool, distinguishing this choice from execution success after the call. [9] StableToolBench instead targets repeatability in large-scale tool learning, using a virtual API server, caching, and a stable evaluator to reduce problems caused by changing API availability. [6] These evaluate complementary capabilities: call judgment and reproducible API interaction.

Interactive environments test a broader level of competence. AgentBench evaluates models across multiple interactive settings, whereas WebArena provides reproducible, realistic website tasks that require reasoning and interaction beyond an isolated function-call test. [18][7] Such tasks measure end-to-end outcomes, but a final success or failure may not explain which intermediate decision caused the result. TRAJECT-Bench addresses that gap by evaluating tool selection, parameterization, and ordering with fine-grained trajectory metrics. [8]

Evaluation itself remains difficult. AgentRewardBench studies automatic judging of web-agent trajectories and notes that rules can be hard to extend and may miss successful trajectories; human evaluation is described as more accurate but slower and costlier, motivating LLM-based alternatives. [19] This creates a second-order measurement problem: conclusions about agent progress depend on whether the evaluator correctly interprets the trajectory, not just whether the agent acted successfully. [19]

## Security and operational controls
A tool-using model can encounter instructions embedded in external material and treat them as relevant to action. InjecAgent studies indirect prompt injections in tool-integrated agents, including goals such as direct harm and private-data exfiltration. [11] AgentDojo provides a dynamic environment for evaluating prompt-injection attacks and defenses in tool-using agents over untrusted data, and emphasizes measuring both task utility and security. [10] These risks are not captured by ordinary task-success scores: an agent may complete a request while also taking an unauthorized action or exposing information. [10][11]

Broader security assessment considers multiple attack surfaces and defenses. Agent Security Bench covers direct and indirect prompt injection, memory poisoning, backdoors, and mixed attacks across prompts, tools, and memory retrieval; its authors report limited effectiveness of existing defenses in their evaluation. [12] ToolEmu takes a complementary testing approach, using an LM-emulated sandbox and evaluator to identify agent failures across high-stakes toolkits. [20] These benchmarks underscore the need to test behavior under adversarial or high-impact conditions before deployment. [12][20]

Operational risk also depends on how much capability an agent is granted. OWASP describes excessive agency as damaging actions arising from unexpected, ambiguous, or manipulated outputs, and identifies excessive functionality, permissions, and autonomy as common root causes. [21] This framing suggests that the agent's available capabilities and authority are part of the safety design, not merely properties of its prompt or model. [21]

## Trends and open problems
The recent trajectory is from demonstrating an action loop toward systematically training, evaluating, and securing tool-using systems. API learning work targets call decisions, arguments, tool coverage, and coherent multi-turn data; benchmark work increasingly separates call selection, stable execution, interactive outcomes, and trajectory quality. [3][4][5][6][8][9] This specialization makes evaluation more diagnostic, but also makes headline comparisons harder when studies measure different parts of the pipeline. [6][7][8][9]

Several problems remain unresolved in the sources reviewed here. First, tool availability and behavior can change, complicating repeatable evaluation even when the agent itself is unchanged. [6] Second, final task completion is an incomplete measure when intermediate actions matter or automated evaluators misjudge trajectories. [19][8] Third, security and utility must be evaluated together: defenses can affect whether useful tasks still succeed, while indirect instructions in tool-accessible content can elicit unsafe behavior. [10][11] Finally, safe deployment requires treating agent capability and permission boundaries as part of the system design: security evaluations report weaknesses in existing defenses, while OWASP identifies excessive functionality, permissions, and autonomy as risk factors. [12][21]

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2022-10-06)
[2] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning. arxiv. https://arxiv.org/abs/2406.05804 (2024-06-09)
[3] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[4] ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. arxiv. https://arxiv.org/abs/2307.16789 (2023-07-31)
[5] ToolFlow: Boosting LLM Tool-Calling Through Natural and Coherent Dialogue Synthesis. arxiv. https://arxiv.org/abs/2410.18447 (2024-10-24)
[6] StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models. arxiv. https://arxiv.org/abs/2403.07714 (2024-03-12)
[7] WebArena: A Realistic Web Environment for Building Autonomous Agents. hf-search. https://huggingface.co/papers/2307.13854 (2023-07-25)
[8] TRAJECT-Bench:A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use. hf-search. https://huggingface.co/papers/2510.04550 (2025-10-06)
[9] When2Call: When (not) to Call Tools. hf-search. https://huggingface.co/papers/2504.18851 (2025-04-26)
[10] AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. web. https://arxiv.org/abs/2406.13352 (n.d.)
[11] InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents. web. https://arxiv.org/abs/2403.02691 (2024-08-04)
[12] Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents. web. https://proceedings.iclr.cc/paper_files/paper/2025/file/5750f91d8fb9d5c02bd8ad2c3b44456b-Paper-Conference.pdf (n.d.)
[13] Large Language Model Agent: A Survey on Methodology, Applications and Challenges. web. http://arxiv.org/abs/2503.21460 (n.d.)
[14] Generative Agents: Interactive Simulacra of Human Behavior. arxiv. https://arxiv.org/abs/2304.03442 (2023-04-07)
[15] Gorilla: Large Language Model Connected with Massive APIs. arxiv. https://arxiv.org/abs/2305.15334 (2023-05-24)
[16] Tool Learning in the Wild: Empowering Language Models as Automatic Tool Agents. arxiv. https://arxiv.org/abs/2405.16533 (2024-05-26)
[17] Querying Databases with Function Calling. arxiv. https://arxiv.org/abs/2502.00032 (2025-01-23)
[18] AgentBench: Evaluating LLMs as Agents. hf-search. https://huggingface.co/papers/2308.03688 (2023-08-07)
[19] AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories. arxiv. https://arxiv.org/abs/2504.08942 (2025-04-11)
[20] IDENTIFYING THE RISKS OF AGENTS WITH AN LM-EMULATED SANDBOX. web. https://proceedings.iclr.cc/paper_files/paper/2024/file/7274ed909a312d4d869cc328ad1c5f04-Paper-Conference.pdf (n.d.)
[21] LLM06:2025 Excessive Agency - OWASP Gen AI Security Project. web. https://genai.owasp.org/llmrisk/llm062025-excessive-agency/ (n.d.)
