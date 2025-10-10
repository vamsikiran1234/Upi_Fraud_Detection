#!/bin/bash

# UPI Fraud Detection - Spark Streaming Job Submission Script
# Submits the streaming job to Spark cluster with optimized configuration

set -e

# Configuration
SPARK_MASTER=${SPARK_MASTER:-"spark://localhost:7077"}
APP_NAME="UPI-Fraud-Detection-Streaming"
MAIN_CLASS="spark_streaming"
JAR_PACKAGES="org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.postgresql:postgresql:42.6.0"

# Spark configuration for streaming workload
SPARK_CONF=(
    "--conf spark.sql.streaming.checkpointLocation=/tmp/spark-checkpoints"
    "--conf spark.sql.streaming.stateStore.providerClass=org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider"
    "--conf spark.sql.adaptive.enabled=true"
    "--conf spark.sql.adaptive.coalescePartitions.enabled=true"
    "--conf spark.serializer=org.apache.spark.serializer.KryoSerializer"
    "--conf spark.sql.streaming.forceDeleteTempCheckpointLocation=true"
    "--conf spark.sql.streaming.minBatchesToRetain=10"
    "--conf spark.sql.streaming.stateStore.maintenanceInterval=60s"
    "--conf spark.sql.streaming.statefulOperator.checkCorrectness.enabled=false"
    "--conf spark.sql.execution.arrow.pyspark.enabled=true"
    "--conf spark.dynamicAllocation.enabled=false"
    "--conf spark.executor.instances=2"
    "--conf spark.executor.cores=2"
    "--conf spark.executor.memory=2g"
    "--conf spark.driver.memory=1g"
    "--conf spark.driver.maxResultSize=512m"
)

# Environment variables
export KAFKA_BROKERS=${KAFKA_BROKERS:-"localhost:9092"}
export INPUT_TOPIC=${INPUT_TOPIC:-"raw-transactions"}
export OUTPUT_TOPIC=${OUTPUT_TOPIC:-"enriched-transactions"}
export AGGREGATION_TOPIC=${AGGREGATION_TOPIC:-"aggregated-features"}
export REDIS_HOST=${REDIS_HOST:-"localhost"}
export REDIS_PORT=${REDIS_PORT:-"6380"}
export POSTGRES_HOST=${POSTGRES_HOST:-"localhost"}
export POSTGRES_PORT=${POSTGRES_PORT:-"5433"}
export POSTGRES_DB=${POSTGRES_DB:-"fraud_detection"}
export POSTGRES_USER=${POSTGRES_USER:-"fraud_user"}
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-"password123"}

echo "🚀 Starting UPI Fraud Detection Streaming Job"
echo "Spark Master: $SPARK_MASTER"
echo "Kafka Brokers: $KAFKA_BROKERS"
echo "Input Topic: $INPUT_TOPIC"
echo "Output Topic: $OUTPUT_TOPIC"

# Submit Spark job
spark-submit \
    --master "$SPARK_MASTER" \
    --name "$APP_NAME" \
    --packages "$JAR_PACKAGES" \
    "${SPARK_CONF[@]}" \
    --py-files spark_streaming.py \
    spark_streaming.py

echo "✅ Streaming job submitted successfully"
