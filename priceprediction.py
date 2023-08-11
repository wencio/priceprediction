from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pymongo

# Inicializar una sesión Spark
spark = SparkSession.builder.appName("PrediccionPrecioCierre").getOrCreate()

# Configuración de conexión a MongoDB
mongo_uri = "mongodb://localhost:27017"  # Cambiar según tu configuración
mongo_database = "nombre_de_la_base_de_datos"
mongo_collection = "nombre_de_la_coleccion"

# Cargar datos desde MongoDB en un DataFrame de Spark
df = spark.read.format("com.mongodb.spark.sql.DefaultSource") \
    .option("uri", mongo_uri) \
    .option("database", mongo_database) \
    .option("collection", mongo_collection) \
    .load()


# Seleccionar las columnas necesarias para el modelo
selected_data = df.select("closing_price", "volume")

# Crear un ensamblador de vectores
vector_assembler = VectorAssembler(inputCols=["volume"], outputCol="features")

# Transformar los datos
assembled_data = vector_assembler.transform(selected_data)

# Dividir el conjunto de datos en entrenamiento y prueba
train_data, test_data = assembled_data.randomSplit([0.8, 0.2], seed=123)

# Crear el modelo de regresión lineal
lr = LinearRegression(featuresCol="features", labelCol="closing_price")

# Ajustar el modelo a los datos de entrenamiento
lr_model = lr.fit(train_data)

# Hacer predicciones en los datos de prueba
predictions = lr_model.transform(test_data)

# Mostrar algunas predicciones y valores reales
predictions.select("prediction", "closing_price").show()

# Detener la sesión Spark
spark.stop()
