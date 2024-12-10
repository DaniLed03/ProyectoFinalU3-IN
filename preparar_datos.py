import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ---------------------- 1. Preparación inicial de los datos ----------------------
# Cargar los datos y eliminar espacios en los nombres de las columnas
file_path = 'exp.xlsx'
df = pd.read_excel(file_path, sheet_name='Respuestas de formulario 1')
df.columns = df.columns.str.strip()

# Separar datos numéricos y textuales
numerical_cols = [
    "¿Cuántas horas al día dedicas al cuidado de tu salud mental (como meditación, ejercicio, o descanso)?",
    "¿Con qué frecuencia sientes estrés relacionado con tus estudios? (veces por semana)",
    "¿Cuántas horas de sueño promedio tienes por noche durante el semestre?",
    "¿Cuántas veces al mes has buscado apoyo psicológico o terapéutico?",
    "¿En qué medida consideras que tu universidad ofrece recursos suficientes para la salud mental? (escala del 1 al 10)"
]
text_cols = [
    "¿Cómo describirías el impacto de la carga académica en tu salud mental?",
    "¿Qué estrategias utilizas para manejar el estrés relacionado con tus estudios?",
    "¿Cómo afecta tu entorno universitario (compañeros, profesores, instalaciones) a tu bienestar emocional?",
    "¿Qué recursos adicionales crees que tu universidad podría ofrecer para mejorar la salud mental de los estudiantes?",
    "¿Qué cambios has notado en tu salud mental desde que comenzaste la universidad, y a qué factores los atribuyes?"
]

numerical_data = df[numerical_cols].dropna()
text_data = df[text_cols].dropna()

# Exportar los datos a un archivo Excel
output_path = 'datos_completos.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    numerical_data.to_excel(writer, sheet_name='Datos Numéricos', index=False)
    text_data.to_excel(writer, sheet_name='Datos Textuales', index=False)

print(f"Datos exportados a '{output_path}'.")

# ---------------------- 2. Análisis exploratorio de datos (AED) ----------------------
# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(numerical_data.describe())

# Visualización de histogramas
numerical_data.hist(bins=10, figsize=(10, 8))
plt.suptitle("Distribución de datos numéricos")
plt.show()

# Diagramas de dispersión
sns.pairplot(numerical_data)
plt.show()

# Nube de palabras
all_text = " ".join(text_data[text_cols[0]].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Nube de palabras: Impacto de la carga académica")
plt.show()

# ---------------------- 3. Aplicación de K-means ----------------------
# Estandarización de datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Determinación del número óptimo de clústeres
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Método del codo")
plt.xlabel("Número de clústeres")
plt.ylabel("Inercia")
plt.show()

# Aplicación de K-means
optimal_k = 3  # Ajusta esto según el gráfico del codo
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Agregar los clústeres al dataframe
numerical_data["Cluster"] = clusters

# Visualización de clústeres
sns.pairplot(numerical_data, hue="Cluster", palette="tab10")
plt.show()

# ---------------------- 4. Aplicación de modelo de clasificación ----------------------
# Preparar datos para clasificación
X = scaled_data
y = clusters  # Etiqueta de clúster para clasificación

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
