
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType
)
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIABETES_SCHEMA = StructType([
    StructField("encounter_id",           IntegerType(), nullable=False),
    StructField("patient_nbr",            IntegerType(), nullable=False),
    StructField("race",                   StringType(),  nullable=True),
    StructField("gender",                 StringType(),  nullable=True),
    StructField("age",                    StringType(),  nullable=True),   # e.g. "[70-80)"
    StructField("weight",                 StringType(),  nullable=True),   # mostly missing
    StructField("admission_type_id",      IntegerType(), nullable=True),
    StructField("discharge_disposition_id", IntegerType(), nullable=True),
    StructField("admission_source_id",    IntegerType(), nullable=True),
    StructField("time_in_hospital",       IntegerType(), nullable=True),
    StructField("payer_code",             StringType(),  nullable=True),
    StructField("medical_specialty",      StringType(),  nullable=True),
    StructField("num_lab_procedures",     IntegerType(), nullable=True),
    StructField("num_procedures",         IntegerType(), nullable=True),
    StructField("num_medications",        IntegerType(), nullable=True),
    StructField("number_outpatient",      IntegerType(), nullable=True),
    StructField("number_emergency",       IntegerType(), nullable=True),
    StructField("number_inpatient",       IntegerType(), nullable=True),
    StructField("diag_1",                 StringType(),  nullable=True),
    StructField("diag_2",                 StringType(),  nullable=True),
    StructField("diag_3",                 StringType(),  nullable=True),
    StructField("number_diagnoses",       IntegerType(), nullable=True),
    StructField("max_glu_serum",          StringType(),  nullable=True),
    StructField("A1Cresult",              StringType(),  nullable=True),
    StructField("metformin",              StringType(),  nullable=True),
    StructField("repaglinide",            StringType(),  nullable=True),
    StructField("nateglinide",            StringType(),  nullable=True),
    StructField("chlorpropamide",         StringType(),  nullable=True),
    StructField("glimepiride",            StringType(),  nullable=True),
    StructField("acetohexamide",          StringType(),  nullable=True),
    StructField("glipizide",              StringType(),  nullable=True),
    StructField("glyburide",              StringType(),  nullable=True),
    StructField("tolbutamide",            StringType(),  nullable=True),
    StructField("pioglitazone",           StringType(),  nullable=True),
    StructField("rosiglitazone",          StringType(),  nullable=True),
    StructField("acarbose",               StringType(),  nullable=True),
    StructField("miglitol",               StringType(),  nullable=True),
    StructField("troglitazone",           StringType(),  nullable=True),
    StructField("tolazamide",             StringType(),  nullable=True),
    StructField("examide",                StringType(),  nullable=True),
    StructField("citoglipton",            StringType(),  nullable=True),
    StructField("insulin",                StringType(),  nullable=True),
    StructField("glyburide_metformin",    StringType(),  nullable=True),
    StructField("glipizide_metformin",    StringType(),  nullable=True),
    StructField("glimepiride_pioglitazone", StringType(), nullable=True),
    StructField("metformin_rosiglitazone", StringType(), nullable=True),
    StructField("metformin_pioglitazone", StringType(),  nullable=True),
    StructField("change",                 StringType(),  nullable=True),
    StructField("diabetesMed",            StringType(),  nullable=True),
    StructField("readmitted",             StringType(),  nullable=True),   # target: NO, >30, <30
])


def create_spark_session(app_name: str = "DiabetesIngestion") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        # In local mode, [*] means use all available CPU cores.
        # In production (EMR, Dataproc, Glue), this config is ignored —
        # the cluster manager allocates resources instead.
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")      # AQE: auto-optimizes joins/shuffles
        .config("spark.sql.shuffle.partitions", "8")       # reduce from default 200 for small data
        .getOrCreate()
    )


def read_raw_csv(spark: SparkSession, input_path: str):
    logger.info(f"Reading raw CSV from: {input_path}")
    df = (
        spark.read
        .option("header", "true")
        .option("nullValue", "?")           # dataset uses "?" for missing values
        .option("mode", "PERMISSIVE")       # log bad rows instead of failing the job
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .schema(DIABETES_SCHEMA)
        .csv(input_path)
    )
    return df


def validate_data(df):
    """
    Lightweight pre-write validation. Fail fast if critical columns are null.

    """
    total_rows = df.count()
    logger.info(f"Total rows read: {total_rows:,}")

    # Check primary key completeness
    null_pks = df.filter(
        F.col("encounter_id").isNull() | F.col("patient_nbr").isNull()
    ).count()

    if null_pks > 0:
        raise ValueError(
            f"CRITICAL: {null_pks} rows have null primary keys. "
            "Investigate source data before proceeding."
        )

    # Check target column completeness
    null_target = df.filter(F.col("readmitted").isNull()).count()
    null_pct = (null_target / total_rows) * 100
    logger.info(f"Null readmitted values: {null_target} ({null_pct:.1f}%)")

    if null_pct > 5:
        logger.warning(
            f"High null rate on target column ({null_pct:.1f}%). "
            "Downstream model quality may be impacted."
        )

    return total_rows


def add_ingestion_metadata(df):
    """
    Add audit columns to every record.
    """
    return df.withColumns({
        "_ingested_at":    F.lit(datetime.utcnow().isoformat()),
        "_source_file":    F.input_file_name(),
        "_pipeline_version": F.lit("1.0.0"),
    })


def write_parquet(df, output_path: str, partition_col: str = "admission_type_id"):
    """
    Write partitioned Parquet to the data lake.
    """
    logger.info(f"Writing Parquet to: {output_path}, partitioned by: {partition_col}")
    (
        df
        # Coalesce reduces the number of output files per partition.
        # Without this, each Spark task writes its own file → many small files.
        .coalesce(4)
        .write
        .mode("overwrite")              # Use "append" for incremental loads
        .partitionBy(partition_col)
        .parquet(output_path)
    )
    logger.info("Write complete.")


def main():
    input_path  = sys.argv[1] if len(sys.argv) > 1 else "data/raw/diabetic_data.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/lake/diabetes_raw"

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")   # reduce Spark's verbose logging

    try:
        df = read_raw_csv(spark, input_path)
        total = validate_data(df)
        df = add_ingestion_metadata(df)
        write_parquet(df, output_path)

        logger.info(f"Ingestion complete. {total:,} rows written to {output_path}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
