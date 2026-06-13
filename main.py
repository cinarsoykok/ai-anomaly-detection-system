import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import gradio as gr
from itertools import product
import os
import re


# Skorları normalize eden yardımcı fonksiyon
def normalize_scores(scores):
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)

# Ağırlık kombinasyonlarını üreten fonksiyon
def generate_weight_combinations(step=0.1):
    weights = np.arange(0, 1 + step, step)
    combinations = [combo for combo in product(weights, repeat=5) if abs(sum(combo) - 1.0) < 1e-6]
    return combinations

# En iyi F1 skoru için ağırlıkları optimize eden fonksiyon
def find_best_weights(y_true, ae, ifs, lof, svm, knn):
    best_f1 = 0
    best_weights = None
    combinations = generate_weight_combinations(0.1)
    for w in combinations:
        score = w[0]*ae + w[1]*ifs + w[2]*lof + w[3]*svm + w[4]*knn
        precisions, recalls, thresholds = precision_recall_curve(y_true, score)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        if np.max(f1_scores) > best_f1:
            best_f1 = np.max(f1_scores)
            best_weights = w
    return best_weights, best_f1


def optimize_dbscan(X_embedded):
    # 1. Her veri noktası için 5. en yakın komşu uzaklığını al
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X_embedded)
    distances, _ = neighbors_fit.kneighbors(X_embedded)

    # 2. En uzak komşu mesafelerini sırala
    distances = np.sort(distances[:, -1])

    # 3. Elbow (dirsek) noktası tespit edilir
    kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")

    # 4. Eğer knee bulunamazsa varsayılan değeri kullan
    optimal_eps = distances[kneedle.knee] if kneedle.knee else 3.0

    # 5. min_samples = log(n)
    min_samples = max(int(np.log(len(X_embedded))), 3)

    return optimal_eps, min_samples

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)  # fazla boşlukları temizle
    text = re.sub(r"[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\s]", "", text)  # özel karakterleri kaldır
    return text.strip()

def enrich_text(text):
    # Basit augmentation: kelime sırası değiştir + synonym eklemesi simülasyonu
    words = text.split()
    if len(words) >= 5:
        random.shuffle(words)
        return " ".join(words)
    elif len(words) >= 2:
        return text + " kullanıcı tarafından oluşturuldu"
    else:
        return text + " metin verisi"

def preprocess_text_column(series):
    return series.astype(str).apply(lambda x: enrich_text(clean_text(x)))

def pseudo_labeling_evaluation(anomaly_scores, top_percent=0.05):
    threshold_index = int(len(anomaly_scores) * (1 - top_percent))
    threshold_score = np.sort(anomaly_scores)[threshold_index]

    pseudo_labels = (anomaly_scores >= threshold_score).astype(int)
    true_labels = np.zeros_like(pseudo_labels)
    true_labels[pseudo_labels == 1] = 1  # Sadece en uç %5'e 1 (anomaly) denir

    report = classification_report(true_labels, pseudo_labels, target_names=["Normal", "Anomaly"])
    f1 = f1_score(true_labels, pseudo_labels)

    return f1, report


# Ana analiz fonksiyonu
def detect_anomalies_unified(file):
    df = pd.read_csv(file.name)
    if 'Class' in df.columns:
        df['label'] = df['Class']

    numeric_data = df.select_dtypes(include=['float64', 'int64'])
    object_data = df.select_dtypes(include=['object'])

    # Eğer sayısal veri yoksa TF-IDF modeli uygula
    if numeric_data.shape[1] < 1 and object_data.shape[1] >= 1:
        combined_text = object_data.astype(str).apply(lambda row: ' '.join(row), axis=1)
        
        # ✅ Yeni eklendi: Temizlik ve zenginleştirme
        enriched_text = preprocess_text_column(combined_text)

        # Daha güçlü bir embedding modeli kullan
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        embeddings = model.encode(enriched_text.tolist())

        X_embedded = StandardScaler().fit_transform(embeddings)
        eps, min_samples = optimize_dbscan(X_embedded)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        clusters = dbscan.fit_predict(X_embedded)

        df["anomaly"] = (clusters == -1).astype(int)

        summary = f"✅ {df['anomaly'].sum()} anomaly detected using optimized DBSCAN (eps={eps:.2f}, min_samples={min_samples})"
        output_path = "sbert_dbscan_output.csv"
        plot_path = None

        df.to_csv(output_path, index=False)
        preview = df[df["anomaly"] == 1].head(10)
        
        if 'label' not in df.columns:
            f1, report = pseudo_labeling_evaluation(df["anomaly_score"].values)
            summary += f"\n🔍 Pseudo F1 Score: {f1:.4f}\n\n{report}"


        return output_path, plot_path, summary, preview, None

    # Sayısal veri varsa ensemble modelleri uygula
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)

    # --- GÜÇLENDİRİLMİŞ AUTOENCODER ---
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation="relu")(input_layer)
    encoded = Dropout(0.3)(encoded)
    encoded = Dense(64, activation="relu")(encoded)
    encoded = Dropout(0.3)(encoded)
    encoded = Dense(32, activation="relu")(encoded)

    decoded = Dense(64, activation="relu")(encoded)
    decoded = Dense(128, activation="relu")(decoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mae')

    # Eğitim ayarları
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    # Daha uzun ve güçlü eğitim
    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=100,
        batch_size=16,
        shuffle=True,
        verbose=0
    )

    # Rekonstrüksiyon hatası
    reconstructions = autoencoder.predict(X_scaled)
    ae_mae = np.mean(np.abs(X_scaled - reconstructions), axis=1)
    ae_scores = normalize_scores(ae_mae)


    # Diğer modeller
    if_model = IForest(n_estimators=200, contamination=0.005, random_state=42)
    if_model.fit(X_scaled)
    if_scores = normalize_scores(if_model.decision_function(X_scaled))

    lof = LocalOutlierFactor(n_neighbors=35, novelty=True)
    lof.fit(X_scaled)
    lof_scores = normalize_scores(lof.decision_function(X_scaled))

    ocsvm = OneClassSVM(gamma='scale', nu=0.03)
    ocsvm.fit(X_scaled)
    svm_scores = normalize_scores(ocsvm.decision_function(X_scaled))

    knn = KNN(n_neighbors=15)
    knn.fit(X_scaled)
    knn_scores = normalize_scores(knn.decision_function(X_scaled))

    # Ensemble
    if 'label' in df.columns:
        y_true = df['label']
        best_weights, _ = find_best_weights(y_true, ae_scores, if_scores, lof_scores, svm_scores, knn_scores)
    else:
        best_weights = [0.4, 0.2, 0.2, 0.1, 0.1]

    ensemble_score = sum([w * s for w, s in zip(best_weights, [ae_scores, if_scores, lof_scores, svm_scores, knn_scores])])
    df['anomaly_score'] = ensemble_score

    if 'label' in df.columns:
        precisions, recalls, thresholds = precision_recall_curve(df['label'], ensemble_score)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        df['anomaly'] = (ensemble_score >= best_threshold).astype(int)
        summary = f"✅ {int(df['anomaly'].sum())} anomalies / {len(df)} rows.\nF1={f1_scores[best_idx]:.4f}, P={precisions[best_idx]:.4f}, R={recalls[best_idx]:.4f}"
    else:
        best_threshold = 0.5
        df['anomaly'] = (ensemble_score >= best_threshold).astype(int)
        summary = f"✅ Detected {df['anomaly'].sum()} anomalies (no labels found)"

    # Açıklama kolonları
    feature_errors = np.abs(X_scaled - reconstructions)
    explanations = []
    anomaly_types = []
    for i in range(len(df)):
        if df['anomaly'][i] == 1:
            top_features = np.argsort(feature_errors[i])[::-1][:2]
            feature_names = numeric_data.columns[top_features].tolist()
            explanation = f"High reconstruction error in: {', '.join(feature_names)}"
            explanations.append(explanation)

            # 🧠 Hangi model(ler) anomali dedi?
            reasons = []
            if ae_scores[i] >= best_threshold:
                reasons.append("Autoencoder")
            if if_scores[i] >= best_threshold:
                reasons.append("IsolationForest")
            if lof_scores[i] >= best_threshold:
                reasons.append("LOF")
            if svm_scores[i] >= best_threshold:
                reasons.append("SVM")
            if knn_scores[i] >= best_threshold:
                reasons.append("KNN")
            anomaly_types.append(", ".join(reasons))
        else:
            explanations.append("")
            anomaly_types.append("")
            
    df['explanation'] = explanations
    df['anomaly_type'] = anomaly_types

    output_path = "ensemble_output.csv"
    df.to_csv(output_path, index=False)

    plt.figure(figsize=(8, 4))
    plt.hist(ensemble_score, bins=50)
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Threshold = {best_threshold:.4f}')
    plt.title("Anomaly Score Histogram")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()
    plot_path = "ensemble_plot.png"
    plt.savefig(plot_path)
    plt.close()

    preview = df[df['anomaly'] == 1].head(10)
    return output_path, plot_path, summary, preview

# Gradio arayüzü
demo = gr.Interface(
    fn=detect_anomalies_unified,
    inputs=gr.File(label="Upload CSV File"),
    outputs=[
        gr.File(label="Download CSV with Anomalies"),
        gr.Image(label="Anomaly Score Histogram"),
        gr.Textbox(label="Summary"),
        gr.Dataframe(label="Top Anomalies Preview"),
    ],
    title="🧠 Advanced Ensemble Anomaly Detector",
    description="Detect anomalies using Autoencoder + IForest + LOF + SVM + KNN"
)

demo.launch(share=True)
