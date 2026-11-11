"""Load the historical 10X-retained specificity cohort without dropping rows."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from ._mhc_pseudo_class_i import MHC_PSEUDO_CLASS_I
from .dataset import DIAGNOSTIC_META_COLUMNS, DropReport, TCRPMHCDataset


@lru_cache(maxsize=1)
def pseudo_dictionary():
    # The release's HLA table matches the original PUFFIN table for this cohort.
    return {
        allele.replace("*", "").replace(":", ""): sequence
        for allele, sequence in MHC_PSEUDO_CLASS_I.items()
    }


class FullCohortDataset(TCRPMHCDataset):
    def __init__(self, csv_path, length_cfg):
        self.csv_path = Path(csv_path)
        self.length_cfg = length_cfg
        self.label_col = "label"
        self.preload = False
        self._cache = None
        self.df = pd.read_csv(self.csv_path, keep_default_na=False, low_memory=False)
        self.df["source_HLA"] = self.df["HLA"].astype(str)
        self.df["HLA"] = self.df["HLA"].astype(str).str.strip().str.upper().map(
            lambda allele: ":".join(allele.split(":")[:2])
        )
        for column in ("Peptide", "CDR1a", "CDR2a", "CDR3a", "CDR1b", "CDR2b", "CDR3b"):
            self.df[column] = self.df[column].astype(str).str.strip().str.upper()
        if "MHC_pseudo_seq" not in self.df:
            self.df["MHC_pseudo_seq"] = ""
        missing = self.df["MHC_pseudo_seq"].astype(str).str.strip().eq("")
        mapping = pseudo_dictionary()
        self.df.loc[missing, "MHC_pseudo_seq"] = self.df.loc[missing, "HLA"].map(
            lambda allele: mapping.get(allele.replace("*", "").replace(":", ""), "X" * 34)
        )
        if "sample_id" not in self.df:
            split = self.csv_path.stem.removeprefix("combined_")
            self.df["sample_id"] = [f"legacy_{split}_{i:07d}" for i in range(len(self.df))]
        columns = [
            column for column in (*DIAGNOSTIC_META_COLUMNS, "source_row_number", "source_HLA")
            if column in self.df
        ]
        self._diagnostic_meta = self.df[columns].astype(str).to_dict("records")
        self.report = DropReport(total_input=len(self.df), total_kept=len(self.df))
