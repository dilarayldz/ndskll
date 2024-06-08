import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
import zipfile

def load_zip_file(zip_file_path, file_name):
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open(file_name) as f:
            return pd.read_csv(f, header=None)

zip_file_path = r"C:\\Users\\dilar\\Downloads\\archive.zip"
train_file_name = 'KDDTrain+.txt'
test_file_name = 'KDDTest+.txt'

df_train = load_zip_file(zip_file_path, train_file_name)
df_test = load_zip_file(zip_file_path, test_file_name)

# Kolon isimlerini ekleyin
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
    'dst_host_srv_rerror_rate', 'class'
]

#ekstra bir kolon varsa
if df_train.shape[1] == 43:
    df_train = df_train.drop(columns=[42])
if df_test.shape[1] == 43:
    df_test = df_test.drop(columns=[42])

# Kolon isimlerini ekleyin
df_train.columns = column_names
df_test.columns = column_names

# Hedef değişken ve özellikleri ayırma
X_train = df_train.drop('class', axis=1)
y_train = df_train['class']

X_test = df_test.drop('class', axis=1)
y_test = df_test['class']

# Kategorik özellikleri numerik değerlere dönüştürme
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col], _ = X_train[col].factorize()
    X_test[col], _ = X_test[col].factorize()

# Öznitelik Seçimi - 16, 14, 8 öznitelik
def select_features(X_train, y_train, X_test, num_features):
    selector = SelectKBest(score_func=f_classif, k=num_features)
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support(indices=True)]
    return X_train_selected, X_test_selected, selected_features

# 16 öznitelik seçimi
X_train_16, X_test_16, selected_features_16 = select_features(X_train, y_train, X_test, 16)
# 14 öznitelik seçimi
X_train_14, X_test_14, selected_features_14 = select_features(X_train, y_train, X_test, 14)
# 8 öznitelik seçimi
X_train_8, X_test_8, selected_features_8 = select_features(X_train, y_train, X_test, 8)

# Seçilen öznitelikleri yazdırma
print("16 Öznitelik ile Seçilenler:")
print(selected_features_16)
print("\n14 Öznitelik ile Seçilenler:")
print(selected_features_14)
print("\n8 Öznitelik ile Seçilenler:")
print(selected_features_8)

# CatBoost modeli ile eğitim ve değerlendirme
def train_evaluate_model(X_train, y_train, X_test, y_test):
    model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, verbose=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Doğruluk Oranı: {accuracy * 100:.2f}%")
    print("Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, zero_division=0))

# 16 öznitelik ile model eğitimi ve değerlendirme
print("16 Öznitelik ile Sonuçlar:")
train_evaluate_model(X_train_16, y_train, X_test_16, y_test)

# 14 öznitelik ile model eğitimi ve değerlendirme
print("\n14 Öznitelik ile Sonuçlar:")
train_evaluate_model(X_train_14, y_train, X_test_14, y_test)

# 8 öznitelik ile model eğitimi ve değerlendirme
print("\n8 Öznitelik ile Sonuçlar:")
train_evaluate_model(X_train_8, y_train, X_test_8, y_test)