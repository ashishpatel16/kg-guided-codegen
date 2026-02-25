def test_import_all_modules():
    # Benchmarks
    import src.benchmarks.dataset_loader

    # Program analysis
    import src.program_analysis.algorithmic_debugger
    import src.program_analysis.file_utils
    import src.benchmarks.triage
    import src.agent.fault_localization.graph
    import src.agent.test_generation.core.generator
