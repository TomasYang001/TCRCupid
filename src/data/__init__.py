"""Data layer: dataset / collate / tokenizer / MHC pseudo mapping.

子模块职责（见 DATA_CONTRACT.md）：
- ``dataset``         : ``TCRPMHCDataset``，读 CSV → dict（步骤 4）
- ``mhc_pseudo``      : ``fill_pseudo(hla)`` / ``has_pseudo(hla)``（步骤 4）
- ``tokenizers``      : ``PMHCESMTokenizer`` + ``CdrTcrTokenizer``（步骤 5–6）
- ``collate``         : ``tcr_epi_collate_fn``，产出 Batch Contract 字段（步骤 7）
- ``generation_dataset`` : positive-only pMHC -> CDR3beta generation dataset
- ``generation_collate`` : pMHC-compatible generation collator and CDR3beta tokenizer
"""
