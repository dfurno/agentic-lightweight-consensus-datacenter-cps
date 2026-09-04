# True-CRP convergent and stress profiles

The benchmark invokes the true persistent `crp_consensus` implementation for `n={5,9,13,25,50,100}`. It reports the convergent fresh-state, one-round profile separately from a frozen stress input that reaches the configured maximum of three rounds for every size. Each row uses 20 warm-ups, 500 latency repetitions, and 20 separate `tracemalloc` repetitions; tracing is disabled during latency measurement.

Convergent median latency ranges from 140.110 to 271.411 microseconds and median traced peak memory from 10,553 to 250,008 bytes. Stress median latency ranges from 166.627 to 215.018 microseconds and median traced peak memory from 10,713 to 247,100 bytes. Every stress input reaches three rounds and is non-convergent under the frozen threshold; median exclusions range from 2 to 96. Timing is empirical and noisy, so these six sizes do not establish a fitted scaling law.

The environment variables requesting one thread were set before NumPy import. Provenance records the request and the 64 logical CPUs visible to the VM separately from the declared 60-core/120-thread host. The result is not an electrical-energy measurement, and no token consumption was measured. The historical pseudo-CRP timing remains classified as FPR-OWA plus snapshot conversion.
