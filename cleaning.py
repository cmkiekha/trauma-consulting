"""
Streamlined Proteomics Data Cleaner

A focused tool for cleaning proteomics data with robust validation:
- Transforms complex proteomics data into analysis-ready format
- Removes unwanted rows and columns
- Fuses protein IDs
- Combines protein labels from different sources
- Validates all steps with detailed checks
- Calculates missingness, outliers, and basic statistics
- Creates visualizations for quality assessment
- Saves intermediate files at each processing step

Usage:
python proteomics_cleaner.py --input data.csv --output-dir results
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
from typing import Dict, List, Tuple, Set
import traceback

#############################################################################
# Setup and Configuration
#############################################################################

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("proteomics_cleaner.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ProteomicsCleaner:
    """
    Streamlined cleaner for proteomics data with validation at each step.

    This class transforms raw proteomics data to a clean format with:
    - Patients as rows
    - Proteins as columns
    - Quality metrics and validation at each step
    """

    def __init__(
        self,
        input_file: str,
        output_dir: str,
        sheet_name=0,
        column_af_index: int = 31,  # Index for column AF
        patient_start_index: int = 47,  # Index for column AV
        quality_threshold: float = 0.5,  # Missing value threshold
    ):
        """
        # Initialize the cleaner with file paths and parameters

        Args:
            input_file: Path to input file (CSV or Excel)
            output_dir: Directory to save output files
            sheet_name: Sheet name or index if Excel file
            column_af_index: Column index for protein labels
            patient_start_index: Column index where patient data starts
            quality_threshold: Threshold for missing values
        """
        # Store parameters
        self.input_file = input_file
        self.output_dir = output_dir
        self.sheet_name = sheet_name
        self.column_af_index = column_af_index
        self.patient_start_index = patient_start_index
        self.quality_threshold = quality_threshold

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize data containers
        self.raw_data = None
        self.af_proteins = []
        self.fused_proteins = {}
        self.patient_columns = {}
        self.patient_data = {}
        self.transformed_data = None
        self.quality_metrics = {}
        self.all_proteins = []

    #############################################################################
    # Data Loading and Extraction Functions
    #############################################################################

    def load_data(self) -> pd.DataFrame:
        """
        # Load the data from CSV or Excel file

        Returns:
            Raw data DataFrame
        """
        logger.info(f"Loading data from {self.input_file}")
        try:
            # Determine file type by extension
            if self.input_file.endswith(".csv"):
                data = pd.read_csv(self.input_file, header=None)
            else:
                data = pd.read_excel(
                    self.input_file, sheet_name=self.sheet_name, header=None
                )
            self.raw_data = data

            # Save raw data
            raw_path = os.path.join(self.output_dir, "01_raw_data.csv")
            data.to_csv(raw_path, index=False)

            logger.info(
                f"Saved raw data to {raw_path} "
                f"({data.shape[0]} rows, {data.shape[1]} columns)"
            )
            print(
                f"Step 1: Loaded raw data "
                f"({data.shape[0]} rows, {data.shape[1]} columns)"
            )

            return data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.error(traceback.format_exc())
            raise

    def extract_protein_labels(self) -> List[str]:
        """
        # Extract protein labels from column AF (rows 4-14)

        Returns:
            List of protein labels
        """
        logger.info("Extracting protein labels from column AF (rows 4-14)")

        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Extract from rows 4-14 (0-indexed as 3-13)
        protein_labels = []
        for row in range(3, 14):
            if row < self.raw_data.shape[0]:
                label = self.raw_data.iloc[row, self.column_af_index]
                if not pd.isna(label):
                    protein_labels.append(str(label))

        # Save protein labels
        labels_df = pd.DataFrame({"ProteinLabel": protein_labels})
        labels_path = os.path.join(self.output_dir, "02_af_protein_labels.csv")
        labels_df.to_csv(labels_path, index=False)

        self.af_proteins = protein_labels

        logger.info(f"Extracted {len(protein_labels)} protein labels from column AF")
        print(f"Step 2: Extracted {len(protein_labels)} protein labels from column AF")

        return protein_labels

    def extract_fused_protein_ids(self) -> Dict:
        """
        # Extract fused protein IDs by combining columns B and X (rows 20+)

        Returns:
            Dictionary mapping row indices to fused protein information
        """
        logger.info("Extracting fused protein IDs from columns B and X (rows 20+)")

        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Process rows starting from row 20 (0-indexed as 19)
        fused_proteins = {}
        for row in range(19, self.raw_data.shape[0]):
            protein_accession = self.raw_data.iloc[row, 1]  # Column B
            modified_sequence = self.raw_data.iloc[row, 23]  # Column X

            if not pd.isna(protein_accession) and not pd.isna(modified_sequence):
                fused_id = f"{protein_accession}_{modified_sequence}"
                fused_proteins[row] = {
                    "row_index": row,
                    "protein_accession": protein_accession,
                    "modified_sequence": modified_sequence,
                    "fused_id": fused_id,
                }

        # Save fused protein IDs with useful metadata
        fused_protein_info = [
            {
                "RowIndex": row + 1,  # 1-indexed for user-friendly viewing
                "ProteinAccession": info["protein_accession"],
                "ModifiedSequence": info["modified_sequence"],
                "FusedID": info["fused_id"],
            }
            for row, info in fused_proteins.items()
        ]

        fused_df = pd.DataFrame(fused_protein_info)
        fused_path = os.path.join(self.output_dir, "03_fused_protein_ids.csv")
        fused_df.to_csv(fused_path, index=False)

        self.fused_proteins = fused_proteins

        logger.info(f"Extracted {len(fused_proteins)} fused protein IDs")
        print(f"Step 3: Extracted {len(fused_proteins)} fused protein IDs")

        return fused_proteins

    def identify_patient_columns(self) -> Dict:
        """
        # Identify patient columns and extract information about patients and timepoints

        Returns:
            Dictionary mapping column indices to patient information
        """
        logger.info("Identifying patient columns starting from column AV")

        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Collect patient information from rows 2-3
        patient_columns = {}
        for col in range(self.patient_start_index, self.raw_data.shape[1]):
            patient_id = self.raw_data.iloc[1, col]
            full_id = self.raw_data.iloc[2, col]

            if not pd.isna(patient_id) and not pd.isna(full_id):
                # Extract timepoint from full_id (e.g., "C-002_day5" -> "day5")
                timepoint = "unknown"
                if "_" in str(full_id):
                    timepoint = str(full_id).split("_")[1]

                patient_columns[col] = {
                    "column_index": col,
                    "patient_id": str(patient_id),
                    "full_id": str(full_id),
                    "timepoint": timepoint,
                }

        # Save patient column information
        patient_info = [
            {
                "ColumnIndex": col,
                "PatientID": info["patient_id"],
                "FullID": info["full_id"],
                "Timepoint": info["timepoint"],
            }
            for col, info in patient_columns.items()
        ]

        patient_df = pd.DataFrame(patient_info)
        patient_path = os.path.join(self.output_dir, "04_patient_columns.csv")
        patient_df.to_csv(patient_path, index=False)

        self.patient_columns = patient_columns

        unique_patients = len(
            set(info["patient_id"] for info in patient_columns.values())
        )
        unique_timepoints = len(
            set(info["timepoint"] for info in patient_columns.values())
        )

        logger.info(
            f"Identified {len(patient_columns)} patient columns "
            f"({unique_patients} unique patients, {unique_timepoints} unique timepoints)"
        )
        print(
            f"Step 4: Identified {len(patient_columns)} patient columns "
            f"({unique_patients} unique patients, {unique_timepoints} unique timepoints)"
        )

        return patient_columns

    def extract_protein_values(self) -> Dict:
        """
        # Extract protein values for all proteins and all patients

        Process:
        1. Extract values for Column AF proteins (rows 4-14)
        2. Extract values for Fused proteins (rows 20+)

        Returns:
            Dictionary with patient data and protein values
        """
        logger.info("Extracting protein values for all proteins")

        # Check prerequisites
        self._check_prerequisites(
            raw_data=True, af_proteins=True, fused_proteins=True, patient_columns=True
        )

        # Create a dictionary to store patient data
        patient_data = {}

        # Process each patient column
        for col_idx, patient_info in self.patient_columns.items():
            patient_id = patient_info["patient_id"]
            timepoint = patient_info["timepoint"]
            key = f"{patient_id}_{timepoint}"

            # Initialize patient entry if not exists
            if key not in patient_data:
                patient_data[key] = {
                    "patient_id": patient_id,
                    "timepoint": timepoint,
                    "values": {},
                }

            # 1. Extract values for AF proteins (rows 4-14)
            for protein_idx, protein_label in enumerate(self.af_proteins):
                row_idx = protein_idx + 3  # rows 4-14 (0-indexed as 3-13)
                if (
                    row_idx < self.raw_data.shape[0]
                    and col_idx < self.raw_data.shape[1]
                ):
                    value = self.raw_data.iloc[row_idx, col_idx]
                    if not pd.isna(value):
                        patient_data[key]["values"][protein_label] = value

            # 2. Extract values for fused proteins (rows 20+)
            for row_idx, protein_info in self.fused_proteins.items():
                fused_id = protein_info["fused_id"]
                if (
                    row_idx < self.raw_data.shape[0]
                    and col_idx < self.raw_data.shape[1]
                ):
                    value = self.raw_data.iloc[row_idx, col_idx]
                    if not pd.isna(value):
                        patient_data[key]["values"][fused_id] = value

        # Save a sample of protein values for validation
        self._save_protein_values_sample(patient_data)

        # Calculate and save protein coverage statistics
        self._calculate_protein_coverage(patient_data)

        self.patient_data = patient_data

        all_proteins = set(self.af_proteins).union(
            {info["fused_id"] for info in self.fused_proteins.values()}
        )
        self.all_proteins = list(all_proteins)

        logger.info(
            f"Extracted values for {len(patient_data)} patient-timepoint combinations "
            f"({len(all_proteins)} proteins)"
        )
        print(
            f"Step 5: Extracted values for {len(patient_data)} patient-timepoint combinations "
            f"({len(all_proteins)} proteins)"
        )

        return patient_data

    def _check_prerequisites(
        self,
        raw_data=False,
        af_proteins=False,
        fused_proteins=False,
        patient_columns=False,
    ):
        """
        # Check if required data is available before proceeding with operations
        """
        if raw_data and self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        if af_proteins and not self.af_proteins:
            raise ValueError(
                "No protein labels extracted. Call extract_protein_labels() first."
            )
        if fused_proteins and not self.fused_proteins:
            raise ValueError(
                "No fused proteins extracted. Call extract_fused_protein_ids() first."
            )
        if patient_columns and not self.patient_columns:
            raise ValueError(
                "No patient columns identified. Call identify_patient_columns() first."
            )

    def _save_protein_values_sample(self, patient_data):
        """
        # Save a sample of protein values for validation
        """
        sample_values = []

        # Get 10 proteins from each source if possible
        af_sample = self.af_proteins[: min(10, len(self.af_proteins))]
        fused_sample = [
            info["fused_id"]
            for _, info in list(self.fused_proteins.items())[
                : min(10, len(self.fused_proteins))
            ]
        ]
        protein_sample = af_sample + fused_sample

        # Get values for these proteins for the first 10 patients
        for i, (patient_key, patient) in enumerate(list(patient_data.items())[:10]):
            for protein in protein_sample:
                if protein in patient["values"]:
                    sample_values.append(
                        {
                            "PatientID": patient["patient_id"],
                            "Timepoint": patient["timepoint"],
                            "Protein": protein,
                            "Value": patient["values"][protein],
                        }
                    )

        sample_df = pd.DataFrame(sample_values)
        sample_path = os.path.join(self.output_dir, "05_protein_values_sample.csv")
        sample_df.to_csv(sample_path, index=False)

    def _calculate_protein_coverage(self, patient_data):
        """
        # Calculate and save protein coverage statistics
        """
        # Calculate protein coverage (% of patients with value for each protein)
        all_proteins = set(self.af_proteins)
        for info in self.fused_proteins.values():
            all_proteins.add(info["fused_id"])

        coverage = {protein: 0 for protein in all_proteins}

        for patient in patient_data.values():
            for protein in patient["values"]:
                coverage[protein] = coverage.get(protein, 0) + 1

        # Convert to percentage
        patient_count = len(patient_data)
        for protein in coverage:
            coverage[protein] = (coverage[protein] / patient_count) * 100

        # Save coverage statistics
        coverage_data = [
            {
                "Protein": protein,
                "CoveragePercent": pct,
                "Source": "AF" if protein in self.af_proteins else "Fused",
            }
            for protein, pct in coverage.items()
        ]

        coverage_df = pd.DataFrame(coverage_data).sort_values(
            "CoveragePercent", ascending=False
        )
        coverage_path = os.path.join(self.output_dir, "06_protein_coverage.csv")
        coverage_df.to_csv(coverage_path, index=False)

    #############################################################################
    # Data Transformation and Analysis Functions
    #############################################################################

    def create_transformed_dataset(
        self, min_coverage_pct: float = 10.0
    ) -> pd.DataFrame:
        """
        # Create the transformed dataset with patients as rows and proteins as columns

        Args:
            min_coverage_pct: Minimum percentage of patients that must have a value
                for a protein to be included

        Returns:
            Transformed DataFrame
        """
        logger.info("Creating transformed dataset")

        if not self.patient_data:
            raise ValueError(
                "No patient data extracted. Call extract_protein_values() first."
            )

        # Calculate coverage for each protein
        all_proteins = set()
        for patient in self.patient_data.values():
            all_proteins.update(patient["values"].keys())

        coverage = {protein: 0 for protein in all_proteins}

        for patient in self.patient_data.values():
            for protein in patient["values"]:
                coverage[protein] = coverage.get(protein, 0) + 1

        # Convert to percentage
        patient_count = len(self.patient_data)
        coverage_pct = {
            protein: (count / patient_count) * 100
            for protein, count in coverage.items()
        }

        # Filter proteins based on coverage
        min_count = (min_coverage_pct / 100) * patient_count
        included_proteins = [
            protein for protein, count in coverage.items() if count >= min_count
        ]

        # Create rows for the transformed dataset
        rows = []

        # Sort patients by ID and timepoint
        sorted_patients = sorted(
            self.patient_data.items(),
            key=lambda x: (x[1]["patient_id"], x[1]["timepoint"]),
        )

        for key, patient in sorted_patients:
            row = {
                "PatientID": patient["patient_id"],
                "Timepoint": patient["timepoint"],
            }

            # Add protein values
            for protein in included_proteins:
                row[protein] = patient["values"].get(protein, None)

            rows.append(row)

        # Create DataFrame
        transformed_df = pd.DataFrame(rows)

        # Save transformed data
        transformed_path = os.path.join(self.output_dir, "07_transformed_data.csv")
        transformed_df.to_csv(transformed_path, index=False)

        # Save list of included proteins
        included_df = pd.DataFrame(
            {
                "Protein": included_proteins,
                "Coverage": [coverage_pct[p] for p in included_proteins],
            }
        ).sort_values("Coverage", ascending=False)

        included_path = os.path.join(self.output_dir, "07_included_proteins.csv")
        included_df.to_csv(included_path, index=False)

        self.transformed_data = transformed_df

        # Log summary
        protein_count = len(included_proteins)
        af_count = sum(1 for p in included_proteins if p in self.af_proteins)
        fused_count = protein_count - af_count

        logger.info(
            f"Created transformed dataset with {transformed_df.shape[0]} rows and {transformed_df.shape[1]} columns"
        )
        logger.info(
            f"Included {protein_count} proteins ({af_count} from AF, {fused_count} fused) "
            f"with ≥{min_coverage_pct}% coverage"
        )

        print(
            f"Step 6: Created transformed dataset with {transformed_df.shape[0]} patients and {protein_count} proteins"
        )
        print(
            f" Included {af_count} proteins from column AF and {fused_count} fused proteins "
            f"with ≥{min_coverage_pct}% coverage"
        )

        return transformed_df

    def calculate_quality_metrics(self) -> Dict:
        """
        # Calculate quality metrics for the transformed dataset

        Metrics include:
        - Missingness percentages
        - Basic statistics
        - Coefficient of variation
        - Outlier detection (both IQR and Z-score based)
        - Flags for problematic proteins

        Returns:
            Dictionary of quality metrics
        """
        logger.info("Calculating quality metrics and detecting outliers")

        if self.transformed_data is None:
            raise ValueError(
                "No transformed data. Call create_transformed_dataset() first."
            )

        # Extract protein columns
        protein_columns = [
            col
            for col in self.transformed_data.columns
            if col not in ["PatientID", "Timepoint"]
        ]

        if not protein_columns:
            raise ValueError("No protein columns found in transformed data")

        # Get protein data subset
        protein_data = self.transformed_data[protein_columns]

        # Calculate basic statistics
        stats, problematic = self._calculate_basic_statistics(protein_data)

        # Calculate outliers
        stats, outlier_data = self._detect_outliers(protein_data, stats, problematic)

        # Save results
        self._save_quality_results(stats, problematic, outlier_data, protein_columns)

        # Create summary and metrics dictionary
        summary = self._create_quality_summary(stats, problematic, protein_columns)

        # Store quality metrics
        self.quality_metrics = {
            "missingness": stats["missing_pct"],
            "statistics": stats,
            "problematic": problematic,
            "summary": summary,
            "outlier_counts_iqr": stats["iqr_outliers"].to_dict(),
            "outlier_counts_z": stats["z_outliers"].to_dict(),
        }

        # Log summary
        logger.info(f"Calculated quality metrics for {len(protein_columns)} proteins")
        logger.info(
            f"Average missingness: {stats['missing_pct'].mean():.2f}%, "
            f"Proteins with >50% missing: {problematic['high_missing'].sum()}"
        )
        logger.info(
            f"Proteins with IQR outliers: {problematic['high_iqr_outliers'].sum()}, "
            f"Z-score outliers: {problematic['high_z_outliers'].sum()}"
        )

        print(f"Step 7: Calculated quality metrics")
        print(f" Average missingness: {stats['missing_pct'].mean():.2f}%")
        print(f" Proteins with >50% missing: {problematic['high_missing'].sum()}")
        print(f" Proteins with high CV (>100%): {problematic['high_cv'].sum()}")
        print(f" Proteins with IQR outliers: {problematic['high_iqr_outliers'].sum()}")
        print(
            f" Proteins with Z-score outliers: {problematic['high_z_outliers'].sum()}"
        )
        print(
            f" Proteins with multiple quality flags: {(problematic['flag_count'] > 1).sum()}"
        )

        return self.quality_metrics

    def _calculate_basic_statistics(self, protein_data):
        """
        # Calculate basic statistics for protein data
        """
        # Calculate missingness per protein
        protein_data = protein_data.apply(pd.to_numeric, errors="coerce")
        missingness = (protein_data.isna().sum() / len(protein_data)) * 100

        # Calculate basic statistics for each protein
        stats = protein_data.describe(
            percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        ).T
        stats["missing_pct"] = missingness

        # Calculate additional statistics
        stats["zero_pct"] = (protein_data == 0).sum() / len(protein_data) * 100
        stats["negative_pct"] = (protein_data < 0).sum() / len(protein_data) * 100
        stats["cv"] = (stats["std"] / stats["mean"] * 100).replace(
            [np.inf, -np.inf], np.nan
        )  # Coefficient of variation

        # Flag problematic proteins
        problematic = pd.DataFrame(index=stats.index)
        problematic["high_missing"] = stats["missing_pct"] > 50
        problematic["high_cv"] = stats["cv"] > 100
        problematic["high_zeros"] = stats["zero_pct"] > 50
        problematic["high_negatives"] = stats["negative_pct"] > 5

        return stats, problematic

    def _detect_outliers(self, protein_data, stats, problematic):
        """
        # Detect outliers using IQR and Z-score methods
        """
        protein_data = protein_data.apply(pd.to_numeric, errors="coerce")

        # Calculate IQR for outlier detection
        stats["iqr"] = stats["75%"] - stats["25%"]
        stats["lower_bound"] = stats["25%"] - 1.5 * stats["iqr"]
        stats["upper_bound"] = stats["75%"] + 1.5 * stats["iqr"]

        # Calculate z-score thresholds (±3 standard deviations)
        stats["z_lower_bound"] = stats["mean"] - 3 * stats["std"]
        stats["z_upper_bound"] = stats["mean"] + 3 * stats["std"]

        # Initialize counters
        stats["iqr_outliers"] = pd.Series(0, index=stats.index)
        stats["z_outliers"] = pd.Series(0, index=stats.index)

        outlier_data = []

        # For each protein, identify outliers
        for protein in stats.index:
            # Skip missing values
            valid_data = protein_data[protein].dropna()
            if len(valid_data) == 0:
                continue

            # Get bounds
            lower = stats.loc[protein, "lower_bound"]
            upper = stats.loc[protein, "upper_bound"]
            z_lower = stats.loc[protein, "z_lower_bound"]
            z_upper = stats.loc[protein, "z_upper_bound"]

            # IQR-based outliers
            iqr_outliers = valid_data[(valid_data < lower) | (valid_data > upper)]
            stats.loc[protein, "iqr_outliers"] = len(iqr_outliers)

            # Z-score-based outliers
            z_outliers = valid_data[(valid_data < z_lower) | (valid_data > z_upper)]
            stats.loc[protein, "z_outliers"] = len(z_outliers)

            # Record outlier information
            self._record_outliers(
                protein,
                iqr_outliers,
                z_outliers,
                lower,
                upper,
                z_lower,
                z_upper,
                outlier_data,
            )

        # Calculate outlier percentages
        stats["iqr_outlier_pct"] = (stats["iqr_outliers"] / len(protein_data)) * 100
        stats["z_outlier_pct"] = (stats["z_outliers"] / len(protein_data)) * 100

        # Add outlier flags
        problematic["high_iqr_outliers"] = stats["iqr_outlier_pct"] > 10
        problematic["high_z_outliers"] = stats["z_outlier_pct"] > 10
        problematic["flag_count"] = problematic.sum(axis=1)

        return stats, outlier_data

    def _record_outliers(
        self,
        protein,
        iqr_outliers,
        z_outliers,
        lower,
        upper,
        z_lower,
        z_upper,
        outlier_data,
    ):
        """
        # Record detailed information about outliers
        """
        # Record IQR outliers
        if len(iqr_outliers) > 0:
            for idx in iqr_outliers.index:
                outlier_row = self.transformed_data.loc[idx]
                outlier_data.append(
                    {
                        "Protein": protein,
                        "PatientTimepoint": f"{outlier_row['PatientID']}_{outlier_row['Timepoint']}",
                        "OutlierType": "IQR",
                        "Value": iqr_outliers[idx],
                        "LowerBound": lower,
                        "UpperBound": upper,
                    }
                )

        # Record Z-score outliers (only if not already recorded as IQR outlier)
        if len(z_outliers) > 0:
            for idx in z_outliers.index:
                # Skip if already recorded as IQR outlier
                if idx in iqr_outliers.index:
                    continue

                outlier_row = self.transformed_data.loc[idx]
                outlier_data.append(
                    {
                        "Protein": protein,
                        "PatientTimepoint": f"{outlier_row['PatientID']}_{outlier_row['Timepoint']}",
                        "OutlierType": "Z-score",
                        "Value": z_outliers[idx],
                        "LowerBound": z_lower,
                        "UpperBound": z_upper,
                    }
                )

    def _save_quality_results(self, stats, problematic, outlier_data, protein_columns):
        """
        # Save quality analysis results to files
        """
        # Save statistics
        stats_path = os.path.join(self.output_dir, "08_protein_statistics.csv")
        stats.to_csv(stats_path)

        # Save problematic proteins
        problematic_path = os.path.join(self.output_dir, "08_problematic_proteins.csv")
        problematic.to_csv(problematic_path)

        # Save detailed outlier information
        if outlier_data:
            outliers_df = pd.DataFrame(outlier_data)
            outliers_path = os.path.join(self.output_dir, "08_protein_outliers.csv")
            outliers_df.to_csv(outliers_path, index=False)

    def _create_quality_summary(self, stats, problematic, protein_columns):
        """
        # Create a summary of quality metrics
        """
        summary = pd.DataFrame(
            {
                "Metric": [
                    "Total patients",
                    "Total proteins",
                    "Avg missing values (%)",
                    "Proteins with >50% missing",
                    "Proteins with high CV (>100%)",
                    "Proteins with >50% zeros",
                    "Proteins with >5% negative values",
                    "Proteins with >10% IQR outliers",
                    "Proteins with >10% Z-score outliers",
                    "Proteins with multiple flags",
                ],
                "Value": [
                    len(self.transformed_data),
                    len(protein_columns),
                    stats["missing_pct"].mean(),
                    problematic["high_missing"].sum(),
                    problematic["high_cv"].sum(),
                    problematic["high_zeros"].sum(),
                    problematic["high_negatives"].sum(),
                    problematic["high_iqr_outliers"].sum(),
                    problematic["high_z_outliers"].sum(),
                    (problematic["flag_count"] > 1).sum(),
                ],
            }
        )

        summary_path = os.path.join(self.output_dir, "08_quality_summary.csv")
        summary.to_csv(summary_path, index=False)

        return summary

    #############################################################################
    # Visualization Functions
    #############################################################################

    def visualize_protein_distributions(self, max_proteins: int = 100) -> str:
        """
        # Create visualizations for protein distributions

        Creates:
        - Overview of quality metrics distributions
        - Individual protein histograms and box plots
        - Interactive HTML index for exploring visualizations

        Args:
            max_proteins: Maximum number of proteins to visualize

        Returns:
            Path to the visualization directory
        """
        logger.info("Creating protein distribution visualizations")

        # Import visualization libraries
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import random
        except ImportError:
            logger.error(
                "Matplotlib and/or Seaborn not installed. Cannot create visualizations."
            )
            print(
                "Error: Matplotlib and/or Seaborn not installed. Cannot create visualizations."
            )
            return None

        # Check prerequisites
        if self.transformed_data is None:
            raise ValueError(
                "No transformed data. Call create_transformed_dataset() first."
            )
        if not hasattr(self, "quality_metrics") or not self.quality_metrics:
            raise ValueError(
                "No quality metrics. Call calculate_quality_metrics() first."
            )

        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Get protein columns and load quality metrics
        protein_columns = [
            col
            for col in self.transformed_data.columns
            if col not in ["PatientID", "Timepoint"]
        ]

        stats = self.quality_metrics["statistics"]
        problematic = self.quality_metrics["problematic"]

        # Create overview plots (distribution of quality metrics)
        self._create_overview_plots(viz_dir, stats, problematic)

        # Select proteins to visualize (prioritizing problematic ones)
        proteins_to_viz = self._select_proteins_to_visualize(
            protein_columns, problematic, max_proteins
        )

        # Create individual protein plots and HTML index
        self._create_individual_protein_plots(
            viz_dir, proteins_to_viz, stats, problematic
        )

        logger.info(f"Created visualizations for {len(proteins_to_viz)} proteins")
        print(
            f"Step 8: Visualizations created for {len(proteins_to_viz)} proteins. See {viz_dir}/index.html"
        )

        return viz_dir

    def _create_overview_plots(self, viz_dir, stats, problematic):
        """
        # Create overview plots of quality metrics distributions
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        # Create distribution overview with 4 subplots
        plt.figure(figsize=(12, 8))

        # Plot distribution of missing percentages
        plt.subplot(2, 2, 1)
        sns.histplot(pd.to_numeric(stats["missing_pct"], errors='coerce'), kde=True)
        plt.title("Distribution of Missing Values (%)")
        plt.xlabel("Missing Percentage")
        plt.ylabel("Count")

        # Plot distribution of CV
        plt.subplot(2, 2, 2)
        sns.histplot(pd.to_numeric(stats["cv"].clip(0, 200), errors='coerce'), kde=True)
        plt.title("Distribution of Coefficient of Variation")
        plt.xlabel("CV (%)")
        plt.ylabel("Count")

        # Plot distribution of IQR outliers
        plt.subplot(2, 2, 3)
        sns.histplot(pd.to_numeric(stats["iqr_outlier_pct"].clip(0, 20), errors='coerce'), kde=True)
        plt.title("Distribution of IQR Outliers (%)")
        plt.xlabel("Outlier Percentage")
        plt.ylabel("Count")

        # Plot flag counts
        plt.subplot(2, 2, 4)
        sns.countplot(x=pd.to_numeric(problematic["flag_count"], errors='coerce'))
        plt.title("Number of Quality Flags per Protein")
        plt.xlabel("Flag Count")
        plt.ylabel("Number of Proteins")

        plt.tight_layout()
        overview_path = os.path.join(viz_dir, "distribution_overview.png")
        plt.savefig(overview_path)
        plt.close()

    def _select_proteins_to_visualize(self, protein_columns, problematic, max_proteins):
        """
        # Select which proteins to visualize, prioritizing problematic ones
        """
        import random

        proteins_to_viz = []

        # Add problematic proteins first (proteins with multiple flags)
        multi_flag_proteins = problematic[problematic["flag_count"] > 1].index.tolist()
        proteins_to_viz.extend(multi_flag_proteins[: min(50, len(multi_flag_proteins))])

        # Add some random proteins if we haven't reached max_proteins
        if len(proteins_to_viz) < max_proteins:
            remaining_proteins = [
                p for p in protein_columns if p not in proteins_to_viz
            ]
            random_proteins = random.sample(
                remaining_proteins,
                min(max_proteins - len(proteins_to_viz), len(remaining_proteins)),
            )
            proteins_to_viz.extend(random_proteins)

        return proteins_to_viz

    def _create_individual_protein_plots(
        self, viz_dir, proteins_to_viz, stats, problematic
    ):
        """
        # Create individual distribution plots for selected proteins and HTML index
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get protein data
        protein_data = self.transformed_data[
            [
                col
                for col in self.transformed_data.columns
                if col not in ["PatientID", "Timepoint"]
            ]
        ].apply(pd.to_numeric, errors='coerce').dropna()

        # Create directory for individual distributions
        distributions_dir = os.path.join(viz_dir, "individual_distributions")
        os.makedirs(distributions_dir, exist_ok=True)

        # Create HTML index start
        html_content = self._create_html_header()

        # Create individual plots
        for i, protein in enumerate(proteins_to_viz):
            if i % 10 == 0:
                logger.info(f"Creating visualizations: {i}/{len(proteins_to_viz)}")

            # Get protein data and skip if empty
            values = protein_data[protein].dropna()
            if len(values) == 0:
                continue

            # Create the plot and save it
            filename = self._create_protein_plot(
                distributions_dir, protein, values, stats
            )

            # Add to HTML content
            html_content.extend(self._create_html_card(protein, filename, problematic))

        # Finish HTML document and write to file
        html_content.extend([" </div>", "</body>", "</html>"])

        index_path = os.path.join(viz_dir, "index.html")
        with open(index_path, "w") as f:
            f.write("\n".join(html_content))

    def _create_html_header(self):
        """
        # Create HTML header for the visualization index
        """
        return [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            " <title>Protein Distribution Visualizations</title>",
            " <style>",
            " body { font-family: Arial, sans-serif; margin: 20px; }",
            " .protein-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }",
            " .protein-card { border: 1px solid #ddd; padding: 10px; border-radius: 5px; }",
            " .protein-card img { max-width: 100%; }",
            " .protein-card h3 { margin-top: 0; }",
            " .flags { color: red; }",
            " .metrics { font-size: 12px; color: #666; margin-bottom: 8px; }",
            " </style>",
            "</head>",
            "<body>",
            " <h1>Protein Distribution Visualizations</h1>",
            " <p>Click on any image to view full size.</p>",
            " <div class='protein-grid'>",
        ]

    def _create_protein_plot(self, distributions_dir, protein, values, stats):
        """
        # Create and save plot for an individual protein
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram with KDE
        sns.histplot(values, kde=True, ax=ax1)
        ax1.set_title(f"{protein} Distribution")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Count")

        # Add IQR and Z-score bounds if available
        if protein in stats.index:
            # IQR bounds
            if "lower_bound" in stats.columns and "upper_bound" in stats.columns:
                lb = stats.loc[protein, "lower_bound"]
                ub = stats.loc[protein, "upper_bound"]
                ax1.axvline(
                    x=lb,
                    color="r",
                    linestyle="--",
                    alpha=0.7,
                    label=f"IQR Lower: {lb:.2f}",
                )
                ax1.axvline(
                    x=ub,
                    color="r",
                    linestyle="--",
                    alpha=0.7,
                    label=f"IQR Upper: {ub:.2f}",
                )

            # Z-score bounds
            if "z_lower_bound" in stats.columns and "z_upper_bound" in stats.columns:
                zlb = stats.loc[protein, "z_lower_bound"]
                zub = stats.loc[protein, "z_upper_bound"]
                ax1.axvline(
                    x=zlb,
                    color="g",
                    linestyle=":",
                    alpha=0.7,
                    label=f"Z Lower: {zlb:.2f}",
                )
                ax1.axvline(
                    x=zub,
                    color="g",
                    linestyle=":",
                    alpha=0.7,
                    label=f"Z Upper: {zub:.2f}",
                )

            ax1.legend()

        # Box plot
        sns.boxplot(y=values, ax=ax2)
        ax2.set_title(f"{protein} Box Plot")
        ax2.set_ylabel("Value")

        # Add text with statistics
        if protein in stats.index:
            stats_text = f"Missing: {stats.loc[protein, 'missing_pct']:.1f}%\n"
            stats_text += f"Mean: {stats.loc[protein, 'mean']:.2f}\n"
            stats_text += f"Std: {stats.loc[protein, 'std']:.2f}\n"
            stats_text += f"CV: {stats.loc[protein, 'cv']:.1f}%\n"

            if "iqr_outliers" in stats.columns:
                stats_text += f"IQR Outliers: {stats.loc[protein, 'iqr_outliers']}\n"
            if "z_outliers" in stats.columns:
                stats_text += f"Z Outliers: {stats.loc[protein, 'z_outliers']}\n"

            ax2.text(1.05, 0.5, stats_text, transform=ax2.transAxes, va="center")

        plt.tight_layout()

        # Save the figure with safe filename
        filename = (
            f"{protein.replace('/', '_').replace('\\', '_').replace(':', '_')}.png"
        )
        filepath = os.path.join(distributions_dir, filename)
        plt.savefig(filepath)
        plt.close()

        return filename

    def _create_html_card(self, protein, filename, problematic):
        """
        # Create HTML card for a protein in the visualization index
        """
        # Get quality flags for this protein
        flags = []
        if protein in problematic.index:
            if problematic.loc[protein, "high_missing"]:
                flags.append("High Missing")
            if problematic.loc[protein, "high_cv"]:
                flags.append("High CV")
            if problematic.loc[protein, "high_zeros"]:
                flags.append("High Zeros")
            if problematic.loc[protein, "high_negatives"]:
                flags.append("Negative Values")
            if (
                "high_iqr_outliers" in problematic.columns
                and problematic.loc[protein, "high_iqr_outliers"]
            ):
                flags.append("IQR Outliers")
            if (
                "high_z_outliers" in problematic.columns
                and problematic.loc[protein, "high_z_outliers"]
            ):
                flags.append("Z Outliers")

        flags_html = f"<div class='flags'>{', '.join(flags)}</div>" if flags else ""

        # Add protein source information
        protein_source = "AF" if protein in self.af_proteins else "Fused"
        source_html = f"<div class='source'>Source: {protein_source}</div>"

        return [
            f" <div class='protein-card'>",
            f" <h3>{protein}</h3>",
            f" {source_html}",
            f" {flags_html}",
            f" <a href='individual_distributions/{filename}' target='_blank'>",
            f" <img src='individual_distributions/{filename}' alt='{protein} distribution'>",
            f" </a>",
            f" </div>",
        ]

    #############################################################################
    # Validation Functions
    #############################################################################

    def validate_transformation(self) -> Dict:
        """
        # Validate the transformation process

        Checks:
        1. Patient preservation - all patients from raw data are in transformed data
        2. Protein representation - proteins from both sources are represented
        3. Value consistency - sample values match between raw and transformed data

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating transformation")

        # Check prerequisites
        if self.raw_data is None or self.transformed_data is None:
            raise ValueError("Raw or transformed data missing. Cannot validate.")

        validation_results = {}

        # 1. Check patient preservation
        validation_results["patient_preservation"] = (
            self._validate_patient_preservation()
        )

        # 2. Check protein representation
        validation_results["protein_preservation"] = (
            self._validate_protein_representation()
        )

        # 3. Check sample values
        validation_results["value_validation"] = self._validate_values()

        # Determine overall validation success
        overall_success = all(
            result["success"] for result in validation_results.values()
        )

        validation_results["overall"] = {
            "success": overall_success,
            "message": (
                "All validation checks passed"
                if overall_success
                else "Some validation checks failed"
            ),
        }

        # Save validation results
        self._save_validation_results(validation_results)

        # Log summary
        logger.info(
            f"Validation completed: {'SUCCESS' if overall_success else 'SOME CHECKS FAILED'}"
        )

        print(f"Step 9: Validation {'PASSED' if overall_success else 'FAILED'}")
        print(
            f" Patient preservation: {validation_results['patient_preservation']['success']}"
        )
        print(
            f" Protein representation: {validation_results['protein_preservation']['success']}"
        )
        print(f" Value validation: {validation_results['value_validation']['success']}")

        return validation_results

    def _validate_patient_preservation(self):
        """
        # Check if all patients from raw data are in the transformed data
        """
        # Get patients from raw data
        patients_in_raw = set()
        for info in self.patient_columns.values():
            patients_in_raw.add(info["patient_id"])

        # Get patients in transformed data
        patients_in_transformed = set(self.transformed_data["PatientID"])

        return {
            "success": len(patients_in_raw) == len(patients_in_transformed),
            "message": f"Expected {len(patients_in_raw)} patients, found {len(patients_in_transformed)}",
            "missing_patients": list(patients_in_raw - patients_in_transformed),
            "extra_patients": list(patients_in_transformed - patients_in_raw),
        }

    def _validate_protein_representation(self):
        """
        # Check if proteins from both sources are represented in the transformed data
        """
        # Get protein columns
        protein_columns = [
            col
            for col in self.transformed_data.columns
            if col not in ["PatientID", "Timepoint"]
        ]

        # Count proteins from each source
        af_in_transformed = sum(1 for p in protein_columns if p in self.af_proteins)
        fused_in_transformed = sum(
            1 for p in protein_columns if p not in self.af_proteins
        )

        return {
            "success": af_in_transformed > 0 and fused_in_transformed > 0,
            "message": f"Found {len(protein_columns)} proteins: {af_in_transformed} from AF, {fused_in_transformed} fused",
            "af_proteins": af_in_transformed,
            "fused_proteins": fused_in_transformed,
        }

    def _validate_values(self):
        """
        # Check if values in transformed data match original values
        """
        # Get protein columns
        protein_columns = [
            col
            for col in self.transformed_data.columns
            if col not in ["PatientID", "Timepoint"]
        ]

        # Sample a few patients and proteins for validation
        sample_patients = list(self.patient_data.items())[
            : min(5, len(self.patient_data))
        ]
        sample_proteins = protein_columns[: min(5, len(protein_columns))]

        value_checks = []
        match_count = 0
        mismatch_count = 0

        for key, patient in sample_patients:
            patient_id = patient["patient_id"]
            timepoint = patient["timepoint"]

            # Find this patient in transformed data
            transformed_row = self.transformed_data[
                (self.transformed_data["PatientID"] == patient_id)
                & (self.transformed_data["Timepoint"] == timepoint)
            ]

            if not transformed_row.empty:
                # Check sample proteins
                for protein in sample_proteins:
                    if (
                        protein in patient["values"]
                        and protein in transformed_row.columns
                    ):
                        original_value = patient["values"][protein]
                        transformed_value = transformed_row[protein].iloc[0]

                        # Check if values match
                        match = self._values_match(original_value, transformed_value)

                        value_checks.append(
                            {
                                "patient_id": patient_id,
                                "timepoint": timepoint,
                                "protein": protein,
                                "original_value": original_value,
                                "transformed_value": transformed_value,
                                "match": match,
                            }
                        )

                        if match:
                            match_count += 1
                        else:
                            mismatch_count += 1

        return {
            "success": mismatch_count == 0,
            "message": f"Checked {match_count + mismatch_count} values: {match_count} match, {mismatch_count} mismatch",
            "samples": value_checks,
        }

    def _values_match(self, value1, value2):
        """
        # Check if two values match, handling various data types and NaN values
        """
        # Both NaN
        if pd.isna(value1) and pd.isna(value2):
            return True

        # One NaN, one not
        if pd.isna(value1) or pd.isna(value2):
            return False

        # Both numerical
        try:
            return np.isclose(float(value1), float(value2))
        except (ValueError, TypeError):
            # Non-numerical values
            return value1 == value2

    def _save_validation_results(self, validation_results):
        """
        # Save validation results to CSV files
        """
        # Save summary validation results
        validation_df = pd.DataFrame(
            [
                {"Check": key, "Success": value["success"], "Message": value["message"]}
                for key, value in validation_results.items()
                if key != "value_validation"
            ]
        )

        validation_path = os.path.join(self.output_dir, "09_validation_results.csv")
        validation_df.to_csv(validation_path, index=False)

        # Save detailed value checks if available
        if (
            "value_validation" in validation_results
            and "samples" in validation_results["value_validation"]
        ):
            value_checks = validation_results["value_validation"]["samples"]
            if value_checks:
                value_checks_df = pd.DataFrame(value_checks)
                value_checks_path = os.path.join(
                    self.output_dir, "09_value_validation_samples.csv"
                )
                value_checks_df.to_csv(value_checks_path, index=False)

    def generate_summary_report(self) -> str:
        """
        # Generate a comprehensive summary report of the cleaning process

        Includes:
        - Dataset overview (patients, proteins, timepoints)
        - Quality metrics summary
        - Problematic proteins identified
        - Files created in the pipeline
        - Next steps for analysis

        Returns:
            Path to the saved report
        """
        logger.info("Generating summary report")

        report_lines = [
            "# Proteomics Data Cleaning Summary",
            f"\nProcessed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nInput file: {self.input_file}",
            f"Output directory: {self.output_dir}",
            "\n## Dataset Summary",
        ]

        if self.transformed_data is not None:
            protein_columns = [
                col
                for col in self.transformed_data.columns
                if col not in ["PatientID", "Timepoint"]
            ]

            report_lines.extend(
                [
                    f"\n- **Patients:** {self.transformed_data.shape[0]}",
                    f"- **Proteins:** {len(protein_columns)}",
                    f"- **Timepoints:** {self.transformed_data['Timepoint'].nunique()}",
                ]
            )

        if self.quality_metrics:
            if "summary" in self.quality_metrics:
                summary = self.quality_metrics["summary"]
                report_lines.append("\n## Quality Summary")

                for _, row in summary.iterrows():
                    metric, value = row["Metric"], row["Value"]

                    # Format value based on type
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)

                    report_lines.append(f"\n- **{metric}:** {formatted_value}")

            if "problematic" in self.quality_metrics:
                problematic = self.quality_metrics["problematic"]
                multi_flag = (problematic["flag_count"] > 1).sum()

                report_lines.append("\n## Potential Problem Areas")
                report_lines.append(
                    f"\n- **Proteins with multiple quality flags:** {multi_flag}"
                )

                if multi_flag > 0:
                    report_lines.append("\n### Top 10 most problematic proteins:")

                    top_problematic = problematic.sort_values(
                        "flag_count", ascending=False
                    ).head(10)

                    for protein in top_problematic.index:
                        flags = []
                        if problematic.loc[protein, "high_missing"]:
                            flags.append("high missing values")
                        if problematic.loc[protein, "high_cv"]:
                            flags.append("high variability")
                        if problematic.loc[protein, "high_zeros"]:
                            flags.append("many zeros")
                        if problematic.loc[protein, "high_negatives"]:
                            flags.append("negative values")
                        if (
                            "high_iqr_outliers" in problematic.columns
                            and problematic.loc[protein, "high_iqr_outliers"]
                        ):
                            flags.append("IQR outliers")
                        if (
                            "high_z_outliers" in problematic.columns
                            and problematic.loc[protein, "high_z_outliers"]
                        ):
                            flags.append("Z-score outliers")

                        # Identify if protein is from AF or fused source
                        protein_source = (
                            "AF" if protein in self.af_proteins else "Fused"
                        )

                        report_lines.append(
                            f"- **{protein}** ({protein_source}): {', '.join(flags)}"
                        )

        # Add file index
        report_lines.extend(
            [
                "\n## Output Files",
                "\n1. **01_raw_data.csv** - Raw data from input file",
                "2. **02_af_protein_labels.csv** - Protein labels from column AF",
                "3. **03_fused_protein_ids.csv** - Fused protein IDs from columns B and X",
                "4. **04_patient_columns.csv** - Mapping of columns to patients",
                "5. **05_protein_values_sample.csv** - Sample of extracted protein values",
                "6. **06_protein_coverage.csv** - Coverage statistics for all proteins",
                "7. **07_transformed_data.csv** - Final transformed dataset",
                "8. **07_included_proteins.csv** - List of proteins included in final dataset",
                "9. **08_protein_statistics.csv** - Statistics for each protein",
                "10. **08_problematic_proteins.csv** - Proteins with quality issues",
                "11. **08_protein_outliers.csv** - Detected outliers by protein and patient",
                "12. **08_quality_summary.csv** - Summary of quality metrics",
                "13. **09_validation_results.csv** - Results of validation checks",
                "14. **09_value_validation_samples.csv** - Validation of sample values",
                "15. **visualizations/** - Visualizations of protein distributions",
            ]
        )

        # Add next steps
        report_lines.extend(
            [
                "\n## Next Steps",
                "\n1. Review the quality metrics to identify problematic proteins",
                "2. Examine the visualizations for proteins with quality flags",
                "3. Consider filtering out proteins with high missingness (>50%)",
                "4. Develop an imputation strategy for proteins with <20% missingness",
                "5. Consider normalization approaches based on data distribution",
                "6. Flag proteins with extreme outliers for special handling",
                "7. Consult domain experts about proteins with unusual distributions",
            ]
        )

        # Save report
        report_path = os.path.join(self.output_dir, "10_cleaning_summary.md")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        logger.info(f"Summary report saved to {report_path}")
        print(f"Step 10: Generated summary report: {report_path}")

        return report_path

    #############################################################################
    # Pipeline Execution
    #############################################################################

    def run_pipeline(
        self, min_coverage_pct: float = 10.0, max_viz_proteins: int = 100
    ) -> bool:
        """
        # Run the complete proteomics data cleaning pipeline with step-by-step validation

        The pipeline includes:
        1. Loading data from CSV/Excel
        2. Extracting protein labels
        3. Extracting fused protein IDs
        4. Identifying patient columns
        5. Extracting protein values
        6. Creating transformed dataset
        7. Calculating quality metrics
        8. Visualizing protein distributions
        9. Validating transformation
        10. Generating summary report

        Args:
            min_coverage_pct: Minimum percentage of patients with value for a protein
                to be included in the final dataset
            max_viz_proteins: Maximum number of proteins to visualize

        Returns:
            True if successful, False otherwise
        """
        try:
            print("\n========== Proteomics Data Cleaning Pipeline ==========\n")

            # Step 1: Load data
            self.load_data()
            input("Press Enter to continue to the next step...")

            # Step 2: Extract protein labels from column AF
            self.extract_protein_labels()
            input("Press Enter to continue to the next step...")

            # Step 3: Extract fused protein IDs
            self.extract_fused_protein_ids()
            input("Press Enter to continue to the next step...")

            # Step 4: Identify patient columns
            self.identify_patient_columns()
            input("Press Enter to continue to the next step...")

            # Step 5: Extract protein values
            self.extract_protein_values()
            input("Press Enter to continue to the next step...")

            # Step 6: Create transformed dataset
            self.create_transformed_dataset(min_coverage_pct=min_coverage_pct)
            input("Press Enter to continue to the next step...")

            # Step 7: Calculate quality metrics
            self.calculate_quality_metrics()
            input("Press Enter to continue to the next step...")

            # Step 8: Visualize protein distributions
            try:
                self.visualize_protein_distributions(max_proteins=max_viz_proteins)
                input("Press Enter to continue to the next step...")
            except Exception as e:
                logger.warning(f"Visualization failed: {e}. Continuing with pipeline.")
                print(f"Warning: Visualization failed: {e}")
                print("Continuing with the pipeline...")
                traceback.print_exc()

            # Step 9: Validate transformation
            self.validate_transformation()
            input("Press Enter to continue to the next step...")

            # Step 10: Generate summary report
            self.generate_summary_report()

            print("\n========== Cleaning pipeline completed successfully! ==========")
            print(
                f"Final transformed dataset saved to: {os.path.join(self.output_dir, '07_transformed_data.csv')}"
            )
            print(
                f"Summary report saved to: {os.path.join(self.output_dir, '10_cleaning_summary.md')}"
            )

            return True

        except Exception as e:
            logger.error(f"Error in cleaning pipeline: {e}")
            logger.error(traceback.format_exc())
            print(f"\nError: {e}")
            print("See log file for more details.")
            return False


#############################################################################
# Command Line Interface
#############################################################################


def parse_arguments():
    """
    # Parse command line arguments for the proteomics cleaner
    """
    parser = argparse.ArgumentParser(
        description="Clean and validate proteomics data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", required=True, help="Input CSV or Excel file")

    parser.add_argument(
        "--output-dir", required=True, help="Output directory for all generated files"
    )

    parser.add_argument(
        "--sheet", default=0, help="Sheet name or index (only used for Excel files)"
    )

    parser.add_argument(
        "--column-af",
        type=int,
        default=31,
        help="Index for column AF containing protein labels",
    )

    parser.add_argument(
        "--patient-start",
        type=int,
        default=47,
        help="Column index where patient data starts",
    )

    parser.add_argument(
        "--min-coverage",
        type=float,
        default=10.0,
        help="Minimum percentage of patients with protein value for inclusion",
    )

    parser.add_argument(
        "--max-viz",
        type=int,
        default=100,
        help="Maximum number of proteins to visualize",
    )

    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Run pipeline without pausing between steps",
    )

    return parser.parse_args()


def main():
    """
    # Main execution function for the proteomics cleaner
    """
    # Parse command line arguments
    args = parse_arguments()

    print(f"\nProteomics Data Cleaner v1.0")
    print(f"Processing file: {args.input}")
    print(f"Output directory: {args.output_dir}")

    # Create cleaner instance
    cleaner = ProteomicsCleaner(
        input_file=args.input,
        output_dir=args.output_dir,
        sheet_name=args.sheet,
        column_af_index=args.column_af,
        patient_start_index=args.patient_start,
    )

    # Modify run_pipeline behavior if batch mode is enabled
    if args.batch_mode:
        # Create a modified version of the run_pipeline method
        original_run_pipeline = cleaner.run_pipeline

        def batch_run_pipeline(*args, **kwargs):
            # Store the original input function
            original_input = __builtins__["input"]

            # Replace input with a function that doesn't require user interaction
            __builtins__["input"] = lambda prompt: None

            try:
                # Run the pipeline with the modified input function
                result = original_run_pipeline(*args, **kwargs)
            finally:
                # Restore the original input function
                __builtins__["input"] = original_input

            return result

        # Replace the run_pipeline method with our batch version
        cleaner.run_pipeline = batch_run_pipeline

    # Run the pipeline
    success = cleaner.run_pipeline(
        min_coverage_pct=args.min_coverage, max_viz_proteins=args.max_viz
    )

    # Return appropriate exit code
    if not success:
        print("\nCleaning pipeline encountered errors. Check the log file for details.")
        return 1

    print("\nProteomic data cleaning completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
