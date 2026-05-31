# V10 Event-Conditioned Kick Policy Development

This walkthrough documents the architectural evolution from the trajectory-locked V3 baseline to the dynamic, event-conditioned V10 kick policy, synthesizing insights from numerous experiments and design iterations.

## 1. The V3 Baseline & Its Limitations
The V3 kick policy achieved a 97% success rate on static balls but was fundamentally limited by its architecture:

*   **Trajectory Memorization**: Trained on only two right-foot kick motions, the LSTM policy essentially memorized the fixed sequence of actions.
*   **Timing Lock**: The policy was rigidly locked to the wall-clock time $t$. In rolling ball tests (V9a, 0.3-1.0 m/s), contact timing collapsed ($C\Delta = +62$ frames). The robot failed to adjust its timing to intercept the ball because the dense tracking rewards forced it to follow the original motion's schedule.
*   **Support Foot Failures**: Attempts to explicitly reward support foot placement, stability, or yaw (V71-V74) resulted in severe degradation. For example, adding "braking" priors caused an 81% miss rate (the robot just stood there), while removing them caused 66% early collisions. Conclusion: motion tracking already implicitly handles support foot control; explicit dense rewards conflict with the tracking rhythm.

## 2. Exploring Solutions: From Targets to Adapters
To address the dynamic ball challenge, several intermediate architectures were proposed and evaluated:

### V10 Target-Conditioned Plan
The initial idea was to provide ball velocity and intercept point observations, retaining the LSTM via partial weight loading. A curriculum of time-aligned rolling balls (where the ball arrives exactly at the nominal kick frame) was designed. However, this didn't resolve the core issue: the dense tracking reward was still pulling the policy toward a fixed reference timeline, preventing true temporal adaptation.

### V10 Phase Adapter
Recognizing the timing bottleneck, a hierarchical "Phase Adapter" was designed. A high-level MLP would output a timing shift ($\Delta\tau$) based on ball history, adjusting the reference motion query to $\tau = t + \Delta\tau$. The V3 low-level policy would remain completely frozen. While conceptually elegant (decoupling timing from the skill execution), training a hierarchical setup with a frozen LSTM presented significant implementation complexities and credit assignment risks within the existing RL framework.

## 3. The Paradigm Shift: V10 Event-Conditioned Kick
The final V10 architecture abandons sequence tracking in favor of **event-conditioned interaction**. The LSTM is replaced with an MLP and explicit history buffers, shifting the paradigm from "tracking a trajectory at time $t$" to "executing a skill based on current phase and conditions."

### Core Architectural Changes
*   **MLP + Flattened History (~420D)**: Replaced the opaque LSTM with a transparent 4-layer MLP (512→256→128→29). Temporal context is explicitly provided via observation history (e.g., joint positions/velocities over the last 3 frames, ball position over 10 frames).
*   **Event-Warped Weak Prior**: Instead of tracking `motion[t]`, the policy tracks `motion[segment, \phi_{event}]`. The reference posture is queried based on the current *semantic event phase* (Approach, Prestrike, Strike, Followthru) rather than wall-clock time. 
*   **Relative Coordinate Focus**: All prior matching is done in ball-relative or kick-direction-relative coordinates. This decouples the skill from absolute world positioning, allowing the robot to adjust its strike geometry dynamically.

### Reward Restructuring
The reward landscape shifted from full-body tracking to interaction-dominant relations:
1.  **Decayed Event-Warped Prior**: Relaxed tracking of the semantically-aligned reference posture, providing a weak baseline without restricting movement.
2.  **Phase-Aware Foot-Ball Relative**: Rewards specific geometric relationships depending on the phase (e.g., swing foot behind ball during prestrike, closing distance during strike).
3.  **Contact Graph Match**: Rewards physical contacts only when they match the expected contacts for the current event phase, heavily penalizing early collisions.

### Training Curriculum & Event Retiming
Because V10 uses a new MLP architecture, it cannot inherit V3's LSTM weights directly. The training pipeline was overhauled:
1.  **BC Warm-Start**: Single-step Behavioral Cloning from V3 teacher rollouts on static balls to initialize the basic kicking mechanics.
2.  **Nominal PPO Recovery**: Fine-tuning with BC regularization to recover 95%+ static kick success under the new observation/reward structure.
3.  **Event Retiming Augmentation**: Randomly scaling the duration of the semantic phases (e.g., shifting the strike window by $\pm 2$ up to $\pm 20$ frames) during training. This forces the policy to decouple from fixed timing and react purely to the event condition inputs.
4.  **Dynamic Ball Adaptation**: Once retiming is mastered, rolling balls with position and arrival time jitter are introduced, relying on the newly acquired timing adaptability.

## Conclusion
The transition to the V10 architecture represents a fundamental shift from **"imitating a trajectory over time"** to **"executing a semantic skill based on environmental conditions"**. By decoupling the reference prior from wall-clock time, flattening the history into a reactive MLP, and enforcing event-based rewards, the policy gains the temporal flexibility required to ultimately solve the dynamic interception challenge.
