# 10. Verifying Conversational Interaction

Previous: [9. Discrete Event Systems](https://app.notion.com/p/9-Discrete-Event-Systems-1cad273a4a0d8032a153eebb65154e1d?pvs=21) 

Next: [11. Dynamical Systems](https://app.notion.com/p/11-Dynamical-Systems-4ec36cae2e40469a85f5aca077a7d1de?pvs=21) 

# Introduction

A significant portion of software development time is spent testing whether a system "works" or not. In HCI, this typically involves usability testing with target users or heuristic evaluation using common-sense guidelines for good user interaction. However, as systems grow in complexity and size, evaluations requiring manual labor—whether from target users or interaction design experts—become a bottleneck. It becomes impossible to test every aspect of the system, and involving human in iterative evaluation becomes prohibitively time-consuming.

Formal methods from computer science offer potential for integrating early verification in the design process, providing effective techniques for testing and reducing testing time (Baier and Katoen, 2008). Formal methods represent a mathematical approach to modeling and analyzing systems. They aim to establish system correctness with mathematical rigor and have been successfully applied in safety-critical systems.

**Model checking**, a formal method for verifying system correctness, uses two key components: a system’s mathematical model and a **temporal logic** specification. The mathematical model such as DTMC represents how the system behaves. And temporal logic formally defines the system's requirements through logical formulas—precise expressions that evaluate to either true or false. These specifications detail what the system should and should not do (Kochenderfer, 2024). Using these components, model checking can automatically and systematically verify whether a given property holds for a finite-state system model.

We discussed mathematical models like DTMCs and MDPs in [9. Discrete Event Systems](https://app.notion.com/p/9-Discrete-Event-Systems-1cad273a4a0d8032a153eebb65154e1d?pvs=21). These models incorporate randomness (e.g., probabilistic transitions) and non-determinism (e.g., user input). In the following sections, we introduce formal specification methods that accommodate these stochastic properties. Specifically, we'll explore a probabilistic specification logic called Probabilistic Computation Tree Logic (PCTL).

## Opportunities for Automated Verification

![image.png](10%20Verifying%20Conversational%20Interaction/image.png)

Although not every aspect of an interface agent's behavior can be automatically verified, several key properties can be evaluated in interactive software (Bolton et al., 2013; LaToza and Myers, 2010), including:

- **Reachability:** LaToza and Myers's study found that programmers' main questions while programming and debugging software systems were about reachability (2010). Reachability testing answers a crucial question: "Can the system get from state A to state B?" In interactive software terms, we might want to verify questions like: "Can a user successfully complete a checkout process?" or "Can the system recover from an error state back to normal operation?"
- **Invariance:** This property ensures that critical system configurations remain stable. For example, in a chatbot, user preferences like language settings should stay constant during conversations unless deliberately modified. Invariance verification confirms that such system parameters maintain their expected values and do not change unexpectedly.

# Temporal Logic

![image.png](10%20Verifying%20Conversational%20Interaction/image%201.png)

Temporal logic is a formal language used to specify and reason about how system behavior evolves over time. It extends propositional logic by incorporating modal and temporal operators. A key application is representing system properties for verification using model checkers.

Traditional temporal logics like CTL and LTL are designed to verify properties of non-probabilistic systems. Given a path consisting of a sequence of states $\omega = s_0 s_1 s_2 \dots$, we can determine whether a temporal property $\phi$ is satisfied. Using temporal logic like CTL, model checking enables us to verify whether there exists a path in a model that satisfies $\phi$, or whether all paths satisfying a given condition meet the specification $\phi$. See more on CTL and LTL in: [Appendix: Formal Logic](https://app.notion.com/p/Appendix-Formal-Logic-1f5d273a4a0d8078ab01c795afca9c0d?pvs=21).

For DTMCs and other probabilistic models, we use **probabilistic model checking,** a probabilistic extension of traditional temporal logics and model checking methods (Baier and Katoen, 2008). These tools verify properties like "The probability of conversion from browsing is at least 70%" or "Less than 20% of users convert without being disengaged.” Tools like [PRISM](https://www.prismmodelchecker.org/manual/PropertySpecification/AllOnOnePage) and [Storm](https://www.stormchecker.org/) calculate the probability of satisfying temporal logic formulas. The logic used here is a probabilistic extension of CTL, called PCTL. 

# PCTL

**Probabilistic Computation Tree Logic (PCTL)** extends CTL by introducing a probabilistic operator. Rather than posing deterministic questions, such as whether there exists a state satisfying a temporal property or whether all states satisfy certain conditions, PCTL enables you to ask: "What is the probability that a given specification is satisfied?" or "Given a probability threshold such as $p=0.5$, does the specification hold?”

## Syntax

$$
\begin{align*}
\phi &::= \text{true} \mid a \mid \phi \land \phi \mid \neg \phi \mid P_{\sim p}[\psi] \\ 

\psi &::= X\phi \mid F\phi \mid G\phi \mid \phi \, U \, \phi \mid \phi \, U^{\leq k} \phi
\end{align*}
$$

Here, 

- $\phi$ is a **state formulae**
- $\psi$ is a **path formulae**

In state formulae, we have:

- $a$ is an **atomic proposition**
- $p \in [0, 1]$ is a probability bound
- $\sim \in \{ <, >, \le, \ge \}$ is used to state quantitative relationship of probability

The temporal operators are defined as follows:

- $X\phi$ (Next): $\phi$ holds in the next state
- $F\phi$ (Finally/Eventually): $\phi$ holds at some point in the future
- $G\phi$ (Globally/Always): $\phi$ holds in all future states
- $\phi_1 \, U \, \phi_2$ (Until): $\phi_1$ holds until $\phi_2$ becomes true
- $\phi_1 \, U^{\leq k} \phi_2$ (Bounded Until): $\phi_1$ holds until $\phi_2$ becomes true within $k \in \mathbb{N}$ steps

PCTL formulas are built from two types of expressions: **state formulae** ($\phi$) and **path formulae** ($\psi$). This two-level structure is fundamental to understanding PCTL. In PCTL, every valid formula must be a state formula—path formulae only appear inside the $P$ operator.

**State formulae** describe properties of individual states:

- `true`: trivially satisfied by every state
- $a$: an atomic proposition (a basic fact about a state, e.g., "converted", "engaged")
- $\phi \land \phi$: conjunction (both properties hold)
- $\neg \phi$: negation (the property does not hold)
- $P_{\sim p}[\psi]$: the probability of path formula $\psi$ being satisfied meets the bound $\sim p$

**Path formulae** describe properties of execution paths (sequences of states):

- These always appear inside the $P$ operator
- They use temporal operators ($X, F, G, U$) to reason about how states evolve over time

### Building Formulas

Let's take a look at some examples of PCTL building blocks and valid PCTL formulae. We use states from the DTMC model that we created in the previous lecture note:

- **Simple atomic proposition:** $\text{engaged}$ would be an atomic proposition indicating that the current state is engaged
- **Negation:** To express that a property is false, we use the negation operator. For example, $\neg \text{abandoned}$ represents that the current state is not abandoned.
- **Next operator:** Path formulae can use temporal operators. For instance, $X \, \text{converted}$ means the next state is converted.
- **Probabilistic query:** A syntactically valid PCTL formula must be a state formula. For example, $P_{\geq 0.5} [ F \, \text{converted} ]$ means "with probability at least 0.5, the system eventually reaches converted." Here, $F \, \text{converted}$ is a path formula, and $P_{\geq 0.5} [ \, \dots \, ]$ is a state formula containing that path formula.
- **Combined formula:** We can combine formulae to represent a complex specification. For instance, $P_{\geq 0.8} [ \neg \text{abandoned} \, U \, \text{engaged} ]$ means "with probability at least 0.8, the system avoids abandoned until becoming engaged."

## Semantics

PCTL semantics define what it means for a state or path to satisfy a formula. We use $s \models \phi$ to mean "state $s$ satisfies formula $\phi$" (or "$\phi$ is true in $s$"). Similarly, $\omega \models \psi$ means "path $\omega$ satisfies path formula $\psi$."

### State Formula Semantics

For a state $s$:

- $s \models \text{true}$ means always holds
- $s \models a$ holds if atomic proposition $a$ is true in state $s$
- $s \models \phi_1 \land \phi_2$ holds if both $\phi_1$ and $\phi_2$ hold in $s$
- $s \models \neg \phi$ holds if $\phi$ does not hold in $s$
- $s \models P_{\sim p}[\psi]$ holds if the probability measure of all paths from $s$ satisfying $\psi$ meets the bound $\sim p$

The key insight is that $P_{\sim p}[\psi]$ transforms a path property into a state property by measuring the likelihood of that path property from a given state.

### Path Formula Semantics

For a path $\omega = s_0 s_1 s_2 \dots$: 

- $\omega \models X\phi$ holds if $s_1 \models \phi$ (the second state satisfies $\phi$)
- $\omega \models F\phi$ holds if there exists some $i \geq 0$ where $s_i \models \phi$ (some state in the path satisfies $\phi$)
- $\omega \models G\phi$ holds if for all $i \geq 0$, $s_i \models \phi$ (every state in the path satisfies $\phi$)
- $\omega \models \phi_1 \, U \, \phi_2$ holds if there exists $j \geq 0$ such that $s_j \models \phi_2$ and for all $i < j$, $s_i \models \phi_1$
- $\omega \models \phi_1 \, U^{\leq k} \phi_2$ holds as above, but $j \leq k$

Consider a user engagement model with states: $\text{browsing, engaged, converted, abandoned}$ like what we saw in the DTMC model from the previous lecture. The formula $s \models P_{< 0.25}[X \, \text{fail}]$ means: "From state $s$, the probability that the next state satisfies 'fail' is less than 0.25."

# Model Checking

Using PCTL, we can describe important properties for interactive systems like probabilistic reachability and invariance:

- **Probabilistic reachability** ($P_{\sim p} [ F \, \phi]$): "What is the probability of eventually reaching a state where $\phi$ holds?" Example: $P_{\geq 0.7}[F \, \text{converted}]$ asks whether users convert with at least 70% probability.
- **Probabilistic invariance** ($P_{\sim p} [G \, \phi]$): "What is the probability that $\phi$ remains true throughout all future states?" Example: $P_{\geq 0.9}[G \, \lnot \text{error}]$ asks whether the system stays error-free with at least 90% probability.

Given such temporal logic specifications and a mathematical model, we can formally verify whether the specifications are true and determine their probability. In this section, we use [Storm](https://www.stormchecker.org/) through its Python interface, [Stormpy](https://moves-rwth.github.io/stormpy/). Stormpy reads models written in the PRISM language, so we can reuse the DTMC and MDP models (`engagement.pm` and `activity_agent.pm`, respectively) and their existing `.pctl` specification files from the previous lecture note.

The `run_stormpy.py` script provides the same two-file workflow as the PRISM command-line program: pass it a model followed by a property file. It accepts files containing comments and multiple properties without requiring semicolons. Run `bash run_stormpy.sh` from this directory to check all six examples. Stormpy does not directly parse PRISM's bounded-global shorthand `G<=k`; the runner translates it to the equivalent `!(F<=k !...)` before model checking.

## DTMC

In DTMCs, transitions are purely probabilistic and there is no nondeterminism. This means that from any state, the system's future behavior is entirely determined by the probability distribution over successor states. When we query a PCTL property on a DTMC, we obtain a single probability value representing the likelihood that the property holds.

The PRISM property language uses the syntax `P=?` to compute the exact probability of a path property. This is called a **quantitative query**; we ask "What is the probability?" rather than "Does the probability meet a threshold?"

### Reachability

Reachability asks: "What is the probability of eventually reaching a state?”

```
// Basic reachability tests the probability of eventually converting
P=? [ F "converted" ]

// Bounded reachability. Tests if the system reaches convert within 10 steps
P=? [ F<=10 "converted" ]
```

Run

```bash
pixi run python run_stormpy.py engagement.pm engagement_reachability.pctl
```

You get:

```
Model: engagement.pm (DTMC, 5 states, 12 transitions)
Properties: 2

(1) P=? [ F "converted" ]
Result: 0.5184

(2) P=? [ F<=10 "converted" ]
Result: 0.511948934783203
```

This result shows:

- `P=? [ F "converted" ]` → 0.518: Starting from the initial state, users have roughly a 52% chance of eventually converting. This is the long-run conversion probability accounting for all possible paths through the system.
- `P=? [ F<=10 "converted" ]` → 0.512: Within 10 steps, the conversion probability is about 51%. The similarity to the unbounded result suggests that most conversions happen relatively quickly—if a user is going to convert, they likely do so within the first 10 interactions.

This provides valuable design insight. If your business requires a 60% conversion rate, this model reveals a gap. You might need to adjust transition probabilities. For example, you could improve the browsing to engaged transition. Or add new states, such as a "re-engagement" intervention.

### Invariance

Invariance asks: "What is the probability of always staying within certain states?”

```
// Invariance. Probability of entering abandoned (failure state)
P=? [ G !"abandoned" ]

// Combining safety and reachability
// Tests the user not going to the failure state without being engaged.
P=? [ !"abandoned" U "engaged" ]
```

Run:

```
pixi run python run_stormpy.py engagement.pm engagement_invariance.pctl
```

You get:

```
Model: engagement.pm (DTMC, 5 states, 12 transitions)
Properties: 2

(1) P=? [ G !"abandoned" ]
Result: 0.5184

(2) P=? [ !"abandoned" U "engaged" ]
Result: 0.7714285714285714
```

This suggests:

- `P=? [ G !"abandoned" ]` → 0.518: There's about a 52% probability of never reaching the abandoned state. This equals the conversion probability because the model has only two terminal states (converted and abandoned); avoiding one means reaching the other.
- `P=? [ !"abandoned" U "engaged" ]` → 0.771: There's a 77% probability of reaching the engaged state while avoiding abandonment. This is higher than the conversion rate because it only requires reaching engagement, not converting afterward.

The gap between engagement probability (77%) and conversion probability (52%) reveals a bottleneck at the engagement→conversion transition. Nearly a quarter of engaged users eventually abandon rather than convert.

### Verification

While `P=?` computes exact probabilities, we often want to verify whether a system meets specific requirements. **Verification queries** use a probability threshold instead of `?`, returning `true` or `false`. This is useful for:

- **Acceptance testing:** Does the system meet minimum performance criteria?
- **Regression testing:** After changes, does the system still satisfy its requirements?
- **Certification:** Can we prove the system meets safety standards?

We specify a probability value for the `P` in place of `?`, like `P>=0.5`.

```
// Is conversion probability at least 50%?
P>=0.5 [ F "converted" ]

// Can we guarantee staying engaged for 5 steps with >30% probability?
P>=0.3 [ G<=5 "engaged" ]

// Is it certain we eventually reach a terminal state?
P>=1 [ F ("converted" | "abandoned") ]
```

Run

```
pixi run python run_stormpy.py engagement.pm engagement_verification.pctl
```

You get

```
Model: engagement.pm (DTMC, 5 states, 12 transitions)
Properties: 3

(1) P>=0.5 [ F "converted" ]
Result: true

(2) P>=0.3 [ G<=5 "engaged" ]
Result: false

(3) P>=1 [ F ("converted" | "abandoned") ]
Result: true
```

The result suggests:

- `P>=0.5 [ F "converted" ]` → true: The system meets the 50% conversion threshold, a potential minimum requirement.
- `P>=0.3 [ G<=5 "engaged" ]` → false: Users do not stay engaged for 5 consecutive steps with 30% probability. Engagement is transient and users quickly move to other states.
- `P>=1 [ F ("converted" | "abandoned") ]` → true: The system always terminates by reaching either converted or abandoned. This confirms there are no infinite loops or deadlocks.

These boolean queries work well in automated testing pipelines. Define a suite of PCTL properties as acceptance criteria and verify them automatically whenever the model changes.

## MDP

MDPs introduce **nondeterminism** alongside probability. In interactive systems, nondeterminism represents choices made by a user giving an input or the environment responding to the interface agent's action. The system designer cannot control these choices but must account for possibilities.

Due to interface agent’s ability to select an action and nondeterminism from the user and the environment, a single PCTL query on an MDP produces a range of probabilities rather than one value. Different resolutions of nondeterministic choices yield different probabilities. Storm computes:

- `Pmax=?`: The maximum probability achievable under the most favorable policy
- `Pmin=?`: The minimum probability under the worst-case resolution

When testing intelligent user interfaces with automated methods, these quantities provide useful insights. `Pmax` represents what's achievable when the user or agent behaves optimally for the property in question. `Pmin` represents the worst case—what happens when choices are made adversarially. The gap between `Pmax` and `Pmin` reveals how much the outcome depends on user or agent behavior versus the system's inherent properties. A large gap suggests the design is sensitive to user behavior. A small gap indicates the outcome is largely determined by the system's probabilistic transitions, regardless of the choices made.

### Reachability

Just as we tested reachability with DTMC models, we can do the same with MDP models. For example:

- $P[F \, \text{success}]$: The fundamental query. "what's the probability of eventually reaching success?" Shows how optimal vs adversarial policies differ.
- $P[ \text{true} \, U^{\le 3} \, s_{success}]$. Bounded reachability. A given state is reachable to $s_{success}$ within three steps.

This translates to PRISM property-language code (found in `activity_agent_reachability.pctl`):

```
// Basic reachability
// what's the probability of eventually reaching success?
Pmax=? [ F "s_success" ]
Pmin=? [ F "s_success" ]

// Bounded reachability
// It asks, can we succeed within 3 steps?
Pmax=? [ F<=3 "s_success" ]
Pmin=? [ F<=3 "s_success" ]

// The above specs are same as:
// Pmax=? [ true U<=3 "s_success" ]
// Pmin=? [ true U<=3 "s_success" ]
```

Run:

```
pixi run python run_stormpy.py activity_agent.pm activity_agent_reachability.pctl
```

You would see the results like:

```
Model: activity_agent.pm (MDP, 13 states, 27 transitions)
Properties: 4

(1) Pmax=? [ F "s_success" ]
Result: 0.6083333333333333

(2) Pmin=? [ F "s_success" ]
Result: 0.5833333333333333

(3) Pmax=? [ F<=3 "s_success" ]
Result: 0.585

(4) Pmin=? [ F<=3 "s_success" ]
Result: 0.42
```

Result suggests:

- **Unbounded reachability (`F "s_success"`):** Pmax = 0.608 (under optimal behavior, success is achievable with 61% probability) and Pmin = 0.583 (under worst-case behavior, success probability is 58%). The 3% gap is small, indicating the outcome depends primarily on probabilistic transitions rather than nondeterministic choices.
- **Bounded reachability (`F<=3 "s_success"`):** Pmax = 0.585 (best-case success within 3 steps) and Pmin = 0.420 (worst-case success within 3 steps). The 17% gap shows that when success happens is more sensitive to choices than whether it happens.

The results suggest that when time limits are not relevant, the agent's policy has little effect. To improve interaction, focus on the model structure and transition probabilities. When there is a time limit—meaning the interaction must complete within a few steps—the agent's policy becomes significant. In this case, finding an optimal policy for agent’s decision making is important.

### Invariance

Invariance is commonly used to specify safety properties. Systems often have undesirable states that should be avoided whenever possible.

- $P \, [G \, \neg s_{abandon}]$ The safety property; "never reach a bad state." Note: for this model, `G !"s_abandon"` equals `F "s_success"` since those are the only terminal states.
- $P[ \neg s_{abandon} \, U \, s_{success}]$ Conditional reachability.  The most practical for HCI — "probability of reaching success *while keeping the user engaged (not reaching the abandon state)*." Combines safety (avoid unwanted states like abandonment) with reachability (reach success).

```
// Basic invariance
// Checks "never reach a bad state"
Pmax=? [ G !"s_abandon" ]
Pmin=? [ G !"s_abandon" ]

// Combining invariance and reachability
// Probability of reaching success while keeping the user away from unwanted state.
Pmax=? [ !"s_abandon" U "s_success" ]
Pmin=? [ !"s_abandon" U "s_success" ]
```

Run

```bash
pixi run python run_stormpy.py activity_agent.pm activity_agent_invariance.pctl
```

You would see

```
Model: activity_agent.pm (MDP, 13 states, 27 transitions)
Properties: 4

(1) Pmax=? [ G !"s_abandon" ]
Result: 0.6083333333333334

(2) Pmin=? [ G !"s_abandon" ]
Result: 0.5833333333333334

(3) Pmax=? [ !"s_abandon" U "s_success" ]
Result: 0.6083333333333333

(4) Pmin=? [ !"s_abandon" U "s_success" ]
Result: 0.5833333333333333
```

- **Basic invariance (`G !"s_abandon"`):** These values (Pmax = 0.608 and Pmin = 0.583) match the reachability results because the model contains only two terminal states; never abandoning is equivalent to eventually succeeding.
- **Combined invariance and reachability (`!"s_abandon" U "s_success"`):** The small difference between these values (Pmax = 0.608, Pmin = 0.583) shows that success and avoiding abandonment are equivalent goals in this model's structure.

In this model, safety (avoiding abandonment) and reaching success are two sides of the same coin. However, in more complex models with multiple failure modes or partial success states, these properties would diverge. Therefore, studying the invariance property of the model provides richer diagnostic information.

### Reward

MDPs can associate **rewards** (or costs) with states and transitions. Rewards quantify aspects of system behavior beyond simple reachability:

- **Interaction cost:** Number of steps to complete a task
- **User effort:** Cognitive or physical effort expended
- **Quality metrics:** Points accumulated for desirable outcomes

The `R` operator computes expected cumulative reward. The query `R=? [ F "done" ]` asks: "What is the expected total reward accumulated before reaching the 'done' state?"

```
// Expected reward to reach terminal done state
Rmax=? [ F "done" ]
Rmin=? [ F "done" ]
```

Run

```bash
pixi run python run_stormpy.py activity_agent.pm activity_agent_reward.pctl
```

You would see

```
Model: activity_agent.pm (MDP, 13 states, 27 transitions)
Properties: 2

(1) Rmax=? [ F "done" ]
Result: 6.083333333333332

(2) Rmin=? [ F "done" ]
Result: 5.833333333333331
```

Under the best policy, users accumulate approximately 6 units of reward before termination, whereas under the worst policy, they accumulate approximately 5.8 units. The small gap (about 0.25) suggests that total reward is relatively insensitive to the policy chosen.

# Discussion

## Model Checking for Interactive Systems

Verification through formal methods and simulation is not very common in HCI. This is perhaps in part due to the success of user-centered design—through rapid prototyping and testing with users and usability experts, researchers can learn a great deal about a system's usability. Another reason is the inherent complexity of verification of interactive systems. Since HCI studies information exchange and interface design, many usability qualities depend not only on software characteristics but also on human sensory, motor, and cognitive capabilities. Verifying these human factors is not trivial. Given these challenges and the scope of what can be formally verified, many researchers opt to conduct user studies and other methods that involve people instead of a model-based verification.

Yet, some researchers predict the field will increasingly adopt verification techniques to assess usability aspects (Murray-Smith et al, 2022). These techniques offer several advantages when used offline during the design process: they reduce the burden on human study participants, enhance testing rigor and reproducibility, ensure testing diversity, and improve safety. They also accelerate design and development while reducing timeline uncertainties. As artificial intelligence improves at perceptual tasks and simulation software better replicates physical environments, we may unlock new possibilities for verifying previously untestable usability properties.

## Validation vs Verification

Let me clarify some difference between the term “verification” and “validation.” The term verification refers to checking model's technical correctness and completeness. When people discuss verification, they focus on confirming that all system requirements have been properly implemented. This process examines whether the system design accurately reflects the specified requirements. This is analogous to unit testing in programming.

Validation operates at a higher level than verification—it determines whether a developed system fulfills its intended purpose and meets user needs. This process is analogous to integration testing in programming. Though validation and verification are distinct concepts, people often use the terms interchangeably. For example, testing with real-world data from user studies, deployment logs, or other sources typically falls under validation, though some might label it verification.

## Mode Confusion and Mental Models

Bolton et al. (2013) reviewed how formal verification applies to Human-Automation Interaction (HAI). They identified **mode confusion** as one of the most prominent applications. Mode confusion occurs when a user's mental model of the system's state diverges from its actual state. This is a common cause of accidents in safety-critical domains like aviation—for example, when a pilot believes the autopilot is in "vertical speed" mode when it's actually in "flight path angle" mode.

Formal methods allow researchers to model both the system (how the device actually works) and the user (a formal model of expected user behavior or mental beliefs). By running model checking on the composition of these two models, we can verify safety properties such as: "Can the system be in Mode A while the user believes it is in Mode B?" If the model checker finds a path to such a state, it has identified a design flaw that causes mode confusion.

## Application to HCI

[David Porfirio, Allison Sauppé, Aws Albarghouthi, Bilge Mutlu (2018) “Authoring and Verifying Human-Robot Interactions”, UIST 2018. Source: [https://youtu.be/hXZwBicPR_E?si=0Mlm2h6ilho50xF4](https://youtu.be/hXZwBicPR_E?si=0Mlm2h6ilho50xF4)](https://youtu.be/hXZwBicPR_E?si=nRWM3NUJpHCOWg5u)

David Porfirio, Allison Sauppé, Aws Albarghouthi, Bilge Mutlu (2018) “Authoring and Verifying Human-Robot Interactions”, UIST 2018. Source: [https://youtu.be/hXZwBicPR_E?si=0Mlm2h6ilho50xF4](https://youtu.be/hXZwBicPR_E?si=0Mlm2h6ilho50xF4)

Formal verification methods could be valuable in HCI contexts where system correctness is not just about functional correctness, but also about ensuring appropriate interaction patterns. Porfirio et al. (2018) demonstrated this in their work on authoring and verifying human-robot interactions. They developed a system allowing designers to create and formally verify interaction sequences for social robots.

# Summary

This lecture introduced probabilistic model checking as a formal verification technique for interactive systems. We covered PCTL as a specification language for expressing probabilistic properties like reachability and invariance, methods for model checking DTMCs and MDPs, and reward analysis for measuring interaction quality. Practical HCI applications include guaranteeing safety properties, identifying design bottlenecks, comparing alternatives, and enabling automated testing, complementing empirical user studies as systems grow more complex.

# Exercises

[Exercise #10.1: PCTL](https://app.notion.com/p/Exercise-10-1-PCTL-30cd273a4a0d80088ae1d0ab297b9746?pvs=21)

# References

- Christel Baier and Joost-Pieter Katoen (2008) Principles of Model Checking
- Matthew L. Bolton, Ellen J. Bass, and Radu I. Siminiceanu (2013) Using Formal Verification to Evaluate Human-Automation Interaction: A Review, IEEE Transactions on SMC: Systems
- Thomas D. LaToza and Brad A. Myers, (2010) Developers Ask Reachability Questions, ICSE 2010
- Mykel J. Kochenderfer, Sydney M. Katz, Anthony L. Corso, and Robert J. Moss (2024) Algorithms for Validation, ([Link](https://algorithmsbook.com/validation/))
- Dave Parker (2011)  Lectures 4 Probabilistic Temporal Logics and 5 PCTL Model Checking for DTMCs, Probabilistic Model Checking Course, University of Oxford

Previous: [9. Discrete Event Systems](https://app.notion.com/p/9-Discrete-Event-Systems-1cad273a4a0d8032a153eebb65154e1d?pvs=21) 

Next: [11. Dynamical Systems](https://app.notion.com/p/11-Dynamical-Systems-4ec36cae2e40469a85f5aca077a7d1de?pvs=21)
