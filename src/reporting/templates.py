METHODOLOGY_TEMPLATE = r"""
\section{Experimental Setup and Methodology}
We evaluate a hierarchical IoT/MAS data-center CPS simulation with sensing agents, edge moderating agents, and an orchestrating agent. Dataset sources are first discovered, downloaded when legally permitted, and audited before any trace is used. Incompatible cyber datasets are used only for pattern-level attack timing and never as thermal measurements.

Each sensor emits $s_i(t)=T_{\mathrm{LHC}}(t)+\epsilon_i(t)+a_i(t)$. Edge aggregation compares direct baselines with FPR-informed OWA consensus using a weak-majority quantifier and deterministic reliability diagnostics. Candidate control actions are proposed by a planner and must be accepted by a deterministic verifier before execution.
"""

DISCUSSION_TEMPLATE = r"""
\section{Discussion}
The framework is positioned as a reproducible CPS/MAS simulation study. FPR-informed OWA is a lightweight robust aggregation mechanism, not a cryptographic or formal Byzantine consensus protocol. Claims are limited to empirical resilience under evaluated deception-like sensor faults. Dataset alignment remains a validity threat whenever substitutes or synthetic fallback traces are used.
"""
