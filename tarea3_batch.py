"""Análisis por lotes de viajes de taxis de Nueva York, enero de 2024."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, hour,
    year, month, round as spark_round
)

spark = (SparkSession.builder
         .appName("Tarea3_BigData_Batch")
         .master("local[2]")
         .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")
os.makedirs("resultados", exist_ok=True)

try:
    # Datos originales
    df = spark.read.parquet("datos/yellow_tripdata_2024-01.parquet")
    print("\nDATOS ORIGINALES")
    df.printSchema()
    df.show(5)

    total_original = df.count()
    print("\nTOTAL DE REGISTROS")
    print(total_original)

    # Limpieza: datos completos, rangos válidos y enero de 2024
    df_limpio = df.dropna(subset=[
        "trip_distance", "total_amount", "passenger_count", "tpep_pickup_datetime"
    ])
    df_limpio = df_limpio.filter(
        (col("trip_distance") > 0) &
        (col("trip_distance") <= 100) &
        (col("total_amount") > 0) &
        (col("total_amount") <= 500) &
        (col("passenger_count") >= 1) &
        (col("passenger_count") <= 6) &
        (year("tpep_pickup_datetime") == 2024) &
        (month("tpep_pickup_datetime") == 1)
    )

    total_limpio = df_limpio.count()
    print("\nREGISTROS VALIDOS")
    print(total_limpio)
    print("\nREGISTROS ELIMINADOS")
    print(total_original - total_limpio)

    print("\nESTADISTICAS GENERALES")
    df_limpio.select(
        spark_round(avg("trip_distance"), 2).alias("Distancia_promedio"),
        spark_round(avg("total_amount"), 2).alias("Valor_promedio")
    ).show()

    # Agrupar viajes por hora
    df_horas = df_limpio.withColumn("hora", hour("tpep_pickup_datetime"))
    viajes_hora = (df_horas.groupBy("hora")
                   .agg(count("*").alias("total_viajes"))
                   .orderBy("hora"))
    print("\nVIAJES POR HORA")
    viajes_hora.show(24)

    viajes_hora.coalesce(1).write.mode("overwrite") \
        .option("header", True).csv("resultados/viajes_por_hora")

    filas = viajes_hora.collect()
    horas = [fila["hora"] for fila in filas]
    totales = [fila["total_viajes"] for fila in filas]

    plt.figure(figsize=(10, 6))
    plt.bar(horas, totales)
    plt.xlabel("Hora del dia")
    plt.ylabel("Numero de viajes")
    plt.title("Analisis de viajes de taxi - Enero 2024")
    plt.xticks(range(24))
    plt.tight_layout()
    plt.savefig("resultados/grafica_viajes.png")
    plt.close()

    print("\nPROCESAMIENTO FINALIZADO")
    print("Grafica guardada correctamente")
    print("Resultados disponibles en carpeta resultados")
finally:
    spark.stop()
