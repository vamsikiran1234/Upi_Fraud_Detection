"""
UPI Fraud Detection - Spark Structured Streaming Job
Real-time feature extraction and enrichment from Kafka streams
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import hashlib

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, lit, udf, window, count, sum as spark_sum, avg, max as spark_max,
    min as spark_min, stddev, lag, lead, row_number, dense_rank, 
    from_json, to_json, struct, explode, split, regexp_extract,
    unix_timestamp, from_unixtime, date_format, hour, dayofweek,
    coalesce, isnan, isnull, broadcast
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, 
    TimestampType, BooleanType, ArrayType, MapType
)
from pyspark.sql.window import Window
import redis
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UPIFraudStreamProcessor:
    """Real-time UPI fraud detection stream processor"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spark = None
        self.redis_client = None
        self.postgres_conn = None
        
    def initialize_spark(self):
        """Initialize Spark session with optimized configuration"""
        self.spark = SparkSession.builder \
            .appName("UPI-Fraud-Detection-Streaming") \
            .config("spark.sql.streaming.checkpointLocation", self.config["checkpoint_location"]) \
            .config("spark.sql.streaming.stateStore.providerClass", 
                   "org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark session initialized successfully")
    
    def initialize_connections(self):
        """Initialize Redis and PostgreSQL connections"""
        # Redis connection
        self.redis_client = redis.Redis(
            host=self.config["redis_host"],
            port=self.config["redis_port"],
            db=self.config["redis_db"],
            decode_responses=True
        )
        
        # PostgreSQL connection
        self.postgres_conn = psycopg2.connect(
            host=self.config["postgres_host"],
            port=self.config["postgres_port"],
            database=self.config["postgres_db"],
            user=self.config["postgres_user"],
            password=self.config["postgres_password"]
        )
        
        logger.info("Database connections initialized successfully")
    
    def get_transaction_schema(self) -> StructType:
        """Define schema for incoming transaction data"""
        return StructType([
            StructField("transaction_id", StringType(), False),
            StructField("upi_id", StringType(), False),
            StructField("amount", DoubleType(), False),
            StructField("merchant_id", StringType(), False),
            StructField("merchant_category", StringType(), False),
            StructField("device_id", StringType(), False),
            StructField("ip_address", StringType(), False),
            StructField("location", StructType([
                StructField("latitude", DoubleType(), False),
                StructField("longitude", DoubleType(), False)
            ]), False),
            StructField("timestamp", TimestampType(), False),
            StructField("payment_method", StringType(), True),
            StructField("session_id", StringType(), True),
            StructField("user_agent", StringType(), True),
            StructField("sms_content", StringType(), True),
            StructField("merchant_notes", StringType(), True)
        ])
    
    def read_kafka_stream(self) -> DataFrame:
        """Read transaction stream from Kafka"""
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.config["kafka_brokers"]) \
            .option("subscribe", self.config["input_topic"]) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .option("maxOffsetsPerTrigger", 10000) \
            .load()
        
        # Parse JSON data
        schema = self.get_transaction_schema()
        parsed_df = df.select(
            from_json(col("value").cast("string"), schema).alias("data"),
            col("timestamp").alias("kafka_timestamp"),
            col("partition"),
            col("offset")
        ).select("data.*", "kafka_timestamp", "partition", "offset")
        
        return parsed_df
    
    def hash_upi_id(self, upi_id: str) -> str:
        """Hash UPI ID for privacy protection"""
        return hashlib.sha256(upi_id.encode()).hexdigest()[:16]
    
    def extract_basic_features(self, df: DataFrame) -> DataFrame:
        """Extract basic transaction features"""
        # Register UDF for hashing
        hash_upi_udf = udf(self.hash_upi_id, StringType())
        
        return df.withColumn("user_hash", hash_upi_udf(col("upi_id"))) \
                 .withColumn("hour", hour(col("timestamp"))) \
                 .withColumn("day_of_week", dayofweek(col("timestamp"))) \
                 .withColumn("is_weekend", when(dayofweek(col("timestamp")).isin([1, 7]), 1).otherwise(0)) \
                 .withColumn("is_night", when((hour(col("timestamp")) >= 22) | (hour(col("timestamp")) <= 6), 1).otherwise(0)) \
                 .withColumn("amount_log", when(col("amount") > 0, log(col("amount"))).otherwise(0)) \
                 .withColumn("merchant_risk_category", 
                            when(col("merchant_category").isin(["gambling", "adult", "crypto"]), "high")
                            .when(col("merchant_category").isin(["grocery", "pharmacy", "utilities"]), "low")
                            .otherwise("medium"))
    
    def extract_velocity_features(self, df: DataFrame) -> DataFrame:
        """Extract velocity-based features using windowing"""
        # Define windows for different time periods
        user_window_1h = Window.partitionBy("user_hash").orderBy("timestamp").rangeBetween(-3600, 0)
        user_window_24h = Window.partitionBy("user_hash").orderBy("timestamp").rangeBetween(-86400, 0)
        device_window_1h = Window.partitionBy("device_id").orderBy("timestamp").rangeBetween(-3600, 0)
        merchant_window_1h = Window.partitionBy("merchant_id").orderBy("timestamp").rangeBetween(-3600, 0)
        
        return df.withColumn("user_txn_count_1h", count("*").over(user_window_1h)) \
                 .withColumn("user_txn_count_24h", count("*").over(user_window_24h)) \
                 .withColumn("user_amount_sum_1h", spark_sum("amount").over(user_window_1h)) \
                 .withColumn("user_amount_sum_24h", spark_sum("amount").over(user_window_24h)) \
                 .withColumn("user_amount_avg_1h", avg("amount").over(user_window_1h)) \
                 .withColumn("user_amount_std_1h", stddev("amount").over(user_window_1h)) \
                 .withColumn("device_txn_count_1h", count("*").over(device_window_1h)) \
                 .withColumn("merchant_txn_count_1h", count("*").over(merchant_window_1h)) \
                 .withColumn("user_unique_merchants_1h", 
                            count("merchant_id").over(user_window_1h)) \
                 .withColumn("user_unique_devices_1h", 
                            count("device_id").over(user_window_1h))
    
    def extract_sequence_features(self, df: DataFrame) -> DataFrame:
        """Extract sequence-based behavioral features"""
        user_window = Window.partitionBy("user_hash").orderBy("timestamp")
        
        return df.withColumn("prev_amount", lag("amount", 1).over(user_window)) \
                 .withColumn("next_amount", lead("amount", 1).over(user_window)) \
                 .withColumn("prev_merchant", lag("merchant_id", 1).over(user_window)) \
                 .withColumn("prev_timestamp", lag("timestamp", 1).over(user_window)) \
                 .withColumn("time_since_last_txn", 
                            when(col("prev_timestamp").isNotNull(),
                                 unix_timestamp("timestamp") - unix_timestamp("prev_timestamp"))
                            .otherwise(0)) \
                 .withColumn("amount_change_ratio",
                            when((col("prev_amount").isNotNull()) & (col("prev_amount") > 0),
                                 col("amount") / col("prev_amount"))
                            .otherwise(1.0)) \
                 .withColumn("is_repeat_merchant",
                            when(col("merchant_id") == col("prev_merchant"), 1).otherwise(0))
    
    def extract_location_features(self, df: DataFrame) -> DataFrame:
        """Extract location-based features"""
        user_location_window = Window.partitionBy("user_hash").orderBy("timestamp").rowsBetween(-10, 0)
        
        return df.withColumn("prev_latitude", lag("location.latitude", 1).over(user_location_window)) \
                 .withColumn("prev_longitude", lag("location.longitude", 1).over(user_location_window)) \
                 .withColumn("location_change",
                            when((col("prev_latitude").isNotNull()) & (col("prev_longitude").isNotNull()),
                                 sqrt(pow(col("location.latitude") - col("prev_latitude"), 2) + 
                                     pow(col("location.longitude") - col("prev_longitude"), 2)))
                            .otherwise(0)) \
                 .withColumn("is_location_jump",
                            when(col("location_change") > 0.1, 1).otherwise(0))  # ~11km threshold
    
    def extract_device_features(self, df: DataFrame) -> DataFrame:
        """Extract device and IP-based features"""
        return df.withColumn("is_mobile_device",
                            when(col("user_agent").contains("Mobile"), 1).otherwise(0)) \
                 .withColumn("browser_type",
                            when(col("user_agent").contains("Chrome"), "chrome")
                            .when(col("user_agent").contains("Firefox"), "firefox")
                            .when(col("user_agent").contains("Safari"), "safari")
                            .otherwise("other")) \
                 .withColumn("ip_class",
                            regexp_extract(col("ip_address"), r"^(\d+)\.", 1).cast(IntegerType())) \
                 .withColumn("is_private_ip",
                            when(col("ip_class").isin([10, 172, 192]), 1).otherwise(0))
    
    def extract_text_features(self, df: DataFrame) -> DataFrame:
        """Extract features from SMS content and merchant notes"""
        # Define suspicious keywords
        suspicious_keywords = ["urgent", "verify", "suspend", "click", "link", "otp", "pin"]
        
        df_with_text = df
        
        # SMS content analysis
        for keyword in suspicious_keywords:
            df_with_text = df_with_text.withColumn(
                f"sms_contains_{keyword}",
                when(col("sms_content").contains(keyword), 1).otherwise(0)
            )
        
        return df_with_text.withColumn("sms_length",
                                      when(col("sms_content").isNotNull(), 
                                           length(col("sms_content"))).otherwise(0)) \
                          .withColumn("sms_word_count",
                                     when(col("sms_content").isNotNull(),
                                          size(split(col("sms_content"), " "))).otherwise(0)) \
                          .withColumn("merchant_notes_length",
                                     when(col("merchant_notes").isNotNull(),
                                          length(col("merchant_notes"))).otherwise(0))
    
    def enrich_with_external_data(self, df: DataFrame) -> DataFrame:
        """Enrich with external data sources (merchant risk scores, IP geolocation, etc.)"""
        # This would typically involve joining with external datasets
        # For now, we'll simulate some enrichment
        
        return df.withColumn("merchant_risk_score",
                            when(col("merchant_risk_category") == "high", 0.8)
                            .when(col("merchant_risk_category") == "medium", 0.5)
                            .otherwise(0.2)) \
                 .withColumn("ip_country", lit("IN"))  # Simplified - would use IP geolocation service
    
    def calculate_risk_indicators(self, df: DataFrame) -> DataFrame:
        """Calculate composite risk indicators"""
        return df.withColumn("velocity_risk",
                            when((col("user_txn_count_1h") > 10) | (col("user_amount_sum_1h") > 100000), 1)
                            .otherwise(0)) \
                 .withColumn("behavioral_risk",
                            when((col("is_night") == 1) & (col("amount") > 50000), 1)
                            .when(col("is_location_jump") == 1, 1)
                            .otherwise(0)) \
                 .withColumn("device_risk",
                            when(col("user_unique_devices_1h") > 3, 1).otherwise(0)) \
                 .withColumn("merchant_risk",
                            when(col("merchant_risk_score") > 0.7, 1).otherwise(0))
    
    def aggregate_features_by_window(self, df: DataFrame) -> DataFrame:
        """Aggregate features using tumbling windows"""
        return df \
            .withWatermark("timestamp", "10 minutes") \
            .groupBy(
                window(col("timestamp"), "5 minutes"),
                col("user_hash")
            ) \
            .agg(
                count("*").alias("txn_count_5min"),
                spark_sum("amount").alias("amount_sum_5min"),
                avg("amount").alias("amount_avg_5min"),
                spark_max("amount").alias("amount_max_5min"),
                spark_sum("velocity_risk").alias("velocity_risk_count_5min"),
                spark_sum("behavioral_risk").alias("behavioral_risk_count_5min")
            )
    
    def write_to_feature_store(self, df: DataFrame, output_mode: str = "append"):
        """Write processed features to feature store (Redis + PostgreSQL)"""
        def write_to_stores(batch_df, batch_id):
            """Write batch to both Redis and PostgreSQL"""
            try:
                # Convert to Pandas for easier manipulation
                pandas_df = batch_df.toPandas()
                
                # Write to Redis (for real-time serving)
                for _, row in pandas_df.iterrows():
                    user_hash = row['user_hash']
                    features = row.to_dict()
                    
                    # Store in Redis with TTL
                    redis_key = f"user_features:{user_hash}"
                    self.redis_client.hset(redis_key, mapping=features)
                    self.redis_client.expire(redis_key, 3600)  # 1 hour TTL
                
                # Write to PostgreSQL (for historical analysis)
                with self.postgres_conn.cursor() as cur:
                    for _, row in pandas_df.iterrows():
                        cur.execute("""
                            INSERT INTO user_features (user_hash, feature_name, feature_value, expires_at)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (user_hash, feature_name) 
                            DO UPDATE SET feature_value = EXCLUDED.feature_value,
                                         created_at = CURRENT_TIMESTAMP
                        """, (
                            row['user_hash'],
                            'realtime_features',
                            json.dumps(row.to_dict()),
                            datetime.utcnow() + timedelta(hours=1)
                        ))
                    self.postgres_conn.commit()
                
                logger.info(f"Batch {batch_id}: Processed {len(pandas_df)} records")
                
            except Exception as e:
                logger.error(f"Error writing batch {batch_id}: {e}")
                self.postgres_conn.rollback()
        
        return df.writeStream \
                 .foreachBatch(write_to_stores) \
                 .outputMode(output_mode) \
                 .option("checkpointLocation", f"{self.config['checkpoint_location']}/feature_store") \
                 .trigger(processingTime='30 seconds')
    
    def write_to_kafka(self, df: DataFrame, topic: str):
        """Write processed data back to Kafka"""
        return df.select(
            col("transaction_id").alias("key"),
            to_json(struct("*")).alias("value")
        ).writeStream \
         .format("kafka") \
         .option("kafka.bootstrap.servers", self.config["kafka_brokers"]) \
         .option("topic", topic) \
         .option("checkpointLocation", f"{self.config['checkpoint_location']}/{topic}") \
         .outputMode("append") \
         .trigger(processingTime='10 seconds')
    
    def run_streaming_pipeline(self):
        """Main streaming pipeline execution"""
        logger.info("Starting UPI Fraud Detection streaming pipeline...")
        
        try:
            # Initialize connections
            self.initialize_spark()
            self.initialize_connections()
            
            # Read from Kafka
            raw_stream = self.read_kafka_stream()
            
            # Feature extraction pipeline
            enriched_stream = raw_stream \
                .transform(self.extract_basic_features) \
                .transform(self.extract_velocity_features) \
                .transform(self.extract_sequence_features) \
                .transform(self.extract_location_features) \
                .transform(self.extract_device_features) \
                .transform(self.extract_text_features) \
                .transform(self.enrich_with_external_data) \
                .transform(self.calculate_risk_indicators)
            
            # Write to feature store
            feature_store_query = self.write_to_feature_store(enriched_stream).start()
            
            # Write enriched data to output topic
            kafka_output_query = self.write_to_kafka(
                enriched_stream, 
                self.config["output_topic"]
            ).start()
            
            # Aggregate features and write to aggregation topic
            aggregated_stream = self.aggregate_features_by_window(enriched_stream)
            aggregation_query = self.write_to_kafka(
                aggregated_stream,
                self.config["aggregation_topic"]
            ).start()
            
            # Wait for termination
            feature_store_query.awaitTermination()
            kafka_output_query.awaitTermination()
            aggregation_query.awaitTermination()
            
        except Exception as e:
            logger.error(f"Streaming pipeline failed: {e}")
            raise
        finally:
            if self.spark:
                self.spark.stop()
            if self.postgres_conn:
                self.postgres_conn.close()

def main():
    """Main entry point"""
    config = {
        "kafka_brokers": os.getenv("KAFKA_BROKERS", "localhost:9092"),
        "input_topic": os.getenv("INPUT_TOPIC", "raw-transactions"),
        "output_topic": os.getenv("OUTPUT_TOPIC", "enriched-transactions"),
        "aggregation_topic": os.getenv("AGGREGATION_TOPIC", "aggregated-features"),
        "checkpoint_location": os.getenv("CHECKPOINT_LOCATION", "/tmp/spark-checkpoints"),
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", "6379")),
        "redis_db": int(os.getenv("REDIS_DB", "0")),
        "postgres_host": os.getenv("POSTGRES_HOST", "localhost"),
        "postgres_port": int(os.getenv("POSTGRES_PORT", "5432")),
        "postgres_db": os.getenv("POSTGRES_DB", "fraud_detection"),
        "postgres_user": os.getenv("POSTGRES_USER", "fraud_user"),
        "postgres_password": os.getenv("POSTGRES_PASSWORD", "password123")
    }
    
    processor = UPIFraudStreamProcessor(config)
    processor.run_streaming_pipeline()

if __name__ == "__main__":
    main()
