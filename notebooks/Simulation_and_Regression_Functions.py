import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from copulas.multivariate import GaussianMultivariate
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# 1. CARGA E IMPUTACIÓN DE DATOS
# ==============================================================================

df = pd.read_csv("datos_onehotenocder.csv")

def imputar_datos_faltantes(df_entrada):
    """
    Limpia el dataset rellenando valores nulos antes de cualquier proceso.
    """
    df_entrada.drop(columns = ["codigoGrasa","idDatosGrasas","Corrosión al Cobre","categoria"],inplace = True)
    df_procesado = df_entrada.copy()
    df_discreto = df_procesado.select_dtypes(include = ['bool'])
    columnas_categoricas = df_discreto.columns
    df_procesado[columnas_categoricas]=df_procesado[columnas_categoricas].to_numpy(dtype=float)
    
    # Imputación Numérica (Media)
    columnas_numericas = df_procesado.select_dtypes(include=[np.number]).columns
    if len(columnas_numericas) > 0:
        imputer_num = SimpleImputer(strategy='mean')
        df_procesado[columnas_numericas] = imputer_num.fit_transform(df_procesado[columnas_numericas])
        
    # Imputación Categórica (Moda)
   
    if len(columnas_categoricas) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_procesado[columnas_categoricas] = imputer_cat.fit_transform(df_procesado[columnas_categoricas])
        
    return df_procesado,list(columnas_categoricas)

# ==============================================================================
# 2. FUNCIÓN DE VECTORIZACIÓN (PRE-SIMULACIÓN)
# ==============================================================================

def vectorizar_texto(df_completo, columnas_texto):
    """
    Transforma las columnas de texto en columnas numéricas (conteo de palabras)
    y elimina las columnas de texto originales. 
    Esto permite que la simulación posterior genere también 'palabras' sintéticas.
    """
    print("--- Vectorizando texto original ---")
    df_vectorizado = df_completo.copy()
    stop_words = [
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como',
    'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta',
    'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos',
    'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes',
    'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus',
    'ellas', 'nosotras', 'vosotros', 'vosotras', 'os', 'mío', 'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos',
    'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'es', 'soy', 'eres', 'somos', 'sois',
    'son', 'sea', 'seamos', 'seáis', 'sean', 'sido', 'estoy', 'estás', 'está', 'estamos', 'estáis', 'están', 'esté', 'estés', 'estemos', 'estéis',
    'estén', 'estaré', 'estarás', 'estará', 'estaremos', 'estaréis', 'estarán', 'estaría', 'estarías', 'estaríamos', 'estaríais', 'estarían',
    'estaba', 'estabas', 'estábamos', 'estabais', 'estaban', 'estuve', 'estuviste', 'estuvo', 'estuvimos', 'estuvisteis', 'estuvieron', 'hubiera',
    'hubieras', 'hubiéramos', 'hubierais', 'hubieran', 'hubiese', 'hubieses', 'hubiésemos', 'hubieseis', 'hubiesen', 'habiendo', 'habido',
    'he', 'has', 'ha', 'hemos', 'habéis', 'han', 'haya', 'hayas', 'hayamos', 'hayáis', 'hayan', 'habré', 'habrás', 'habrá', 'habremos',
    'habréis', 'habrán', 'habría', 'habrías', 'habríamos', 'habríais', 'habrían', 'había', 'habías', 'habíamos', 'habíais', 'habían', 'hube',
    'hubiste', 'hubo', 'hubimos', 'hubisteis', 'hubieron', 'ten', 'tengo', 'tienes', 'tiene', 'tenemos', 'tenéis', 'tienen', 'tenga', 'tengas',
    'tengamos', 'tengáis', 'tengan', 'tendré', 'tendrás', 'tendrá', 'tendremos', 'tendréis', 'tendrán', 'tendría', 'tendrías', 'tendríamos',
    'tendríais', 'tendrían', 'tenía', 'tenías', 'teníamos', 'teníais', 'tenían', 'tuve', 'tuviste', 'tuvo', 'tuvimos', 'tuvisteis', 'tuvieron'
]
    
    # Usamos un vectorizador común para mantener consistencia
    vectorizer = CountVectorizer(max_features=10, stop_words=stop_words,binary=True)
    
    for col in columnas_texto:
        # Asegurar string
        textos = df_vectorizado[col].astype(str)
        
        # Ajustar y transformar
        matrix = vectorizer.fit_transform(textos)
        
        # Nombres de nuevas columnas
        feature_names = vectorizer.get_feature_names_out()
        new_cols = [f"{col}_{word}" for word in feature_names]
        
        # Crear DataFrame con las nuevas features
        df_feats = pd.DataFrame(matrix.toarray(), columns=new_cols, index=df_vectorizado.index)
        
        # Concatenar y eliminar columna de texto original
        df_vectorizado = pd.concat([df_vectorizado, df_feats], axis=1)
        df_vectorizado.drop(columns=[col], inplace=True)
        
    print(f"Texto vectorizado. Columnas totales ahora: {df_vectorizado.shape[1]}")
    return df_vectorizado

# ==============================================================================
# 3. MÉTODOS DE SIMULACIÓN (NUMÉRICA)
# ==============================================================================

def simular_datos_knn_numerico(df_numerico, n_nuevos=10000, k=5):
    """Simula datos continuos usando interpolación KNN."""
    X = df_numerico.to_numpy(dtype=float)
    n, d = X.shape
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xn = (X - mean) / std

    nuevos = []
    for _ in range(n_nuevos):
        base_idx = np.random.randint(0, n)
        base = Xn[base_idx]
        dist = np.sqrt(np.sum((Xn - base) ** 2, axis=1))
        vecinos_idx = np.argsort(dist)[1:k+1]
        vecino = Xn[np.random.choice(vecinos_idx)]
        nuevo = base + np.random.rand() * (vecino - base)
        nuevos.append(nuevo)

    return pd.DataFrame(np.array(nuevos) * std + mean, columns=df_numerico.columns)

def simular_datos_copula(df_numerico, n_nuevos=100):
    """Simula datos continuos usando Cópula Gaussiana."""
    model = GaussianMultivariate()
    model.fit(df_numerico)
    return model.sample(n_nuevos)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def train_vae_and_generate(df_numerico, n_nuevos=100, epochs=500):
    """Simula datos continuos usando VAE."""
    data = df_numerico.values.astype(float)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    tensor_x = torch.Tensor(data_scaled)
    dataloader = DataLoader(TensorDataset(tensor_x), batch_size=32, shuffle=True)
    
    vae = VAE(data.shape[1])
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    for _ in range(epochs):
        for batch in dataloader:
            x_b = batch[0]
            optimizer.zero_grad()
            recon, mu, logvar = vae(x_b)
            loss = nn.functional.mse_loss(recon, x_b, reduction='sum') - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss.backward()
            optimizer.step()
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_nuevos, 10)
        gen = vae.decoder(z).numpy()
    return pd.DataFrame(scaler.inverse_transform(gen), columns=df_numerico.columns)

def choose_generation_numeric(metodo, df, n):
    if metodo == 'knn': return simular_datos_knn_numerico(df, n)
    elif metodo == 'gaussian': return simular_datos_copula(df, n)
    elif metodo == 'vae': return train_vae_and_generate(df, n)
    return pd.DataFrame()

# ==============================================================================
# 4. SIMULACIÓN CATEGÓRICA Y DE TEXTO (KNN)
# ==============================================================================

def simular_datos_knn_categoricos_y_texto(df_discreto, n_nuevos=100, k=5):
    """
    Genera datos para variables categóricas (One-Hot) y Texto Vectorizado.
    Utiliza KNN con promedio y redondeo (simulando votación) para mantener
    la naturaleza entera/binaria de los datos.
    
    Parámetros:
    -----------
    df_discreto : pd.DataFrame
        DataFrame con columnas One-Hot (0/1) y Counts (0,1,2...).
    """
    X = df_discreto.to_numpy(dtype=float)
    n_muestras = X.shape[0]
    nuevos = []
    
    for _ in range(n_nuevos):
        # 1. Punto base
        idx_base = np.random.randint(0, n_muestras)
        base = X[idx_base]
        
        # 2. Vecinos
        dist = np.sqrt(np.sum((X - base) ** 2, axis=1))
        vecinos_idx = np.argsort(dist)[1:k+1]
        vecinos_valores = X[vecinos_idx]
        
        # 3. Promedio y Redondeo (Votación para enteros)
        # Esto asegura que si los vecinos tienen 0 y 1, salga 0 o 1.
        # Si tienen conteos de palabras 2, 2, 3 -> saldrá 2.
        nuevo_punto = np.round(np.mean(vecinos_valores, axis=0))
        
        nuevos.append(nuevo_punto)
        
    # Convertimos a int porque son categorías o conteos de palabras
    return pd.DataFrame(nuevos, columns=df_discreto.columns).astype(int)

def generar_dataset_con_vectorizacion_simulada(df_vectorizado, config_cantidades,categoric_cols):
    """
    Coordina la simulación:
    1. Separa numéricos continuos vs discretos (Categorías + Texto Vectorizado).
    2. Simula continuos con el método elegido (VAE/Gaussian/KNN).
    3. Simula discretos SIEMPRE con KNN .
    4. Une los resultados.
    """
    # 1. Identificar columnas discretas (binarias o enteros de texto)
    # Criterio: Si son object, bool, o si son numéricas pero parecen conteos/binarias
    # En este dataset procesado, 'One-Hot' son 0/1 y 'Texto' son enteros >= 0.
    
    # Vamos a asumir que las columnas originales numéricas del dataset físico
    # son las continuas, y el resto (one-hot y las nuevas vectorizadas) son discretas.
    # Una heurística robusta: columnas con pocos valores únicos o enteros.
    
    # Para simplificar y ser precisos:
    # Las columnas generadas por vectorizer contienen "_" (ej: descripcion_oil)
   
    cols_discretas = categoric_cols
    cols_continuas = [c for c in df_vectorizado.columns if c not in cols_discretas]
    
    df_cont = df_vectorizado[cols_continuas]
    df_disc = df_vectorizado[cols_discretas]
    
    print(f"Separación para simulación: {df_cont.shape[1]} Continuas | {df_disc.shape[1]} Discretas (Cat+Texto)")
    
    lista_cont = [df_cont]
    lista_disc = [df_disc]
    
    for metodo, n in config_cantidades.items():
        if n > 0:
            print(f"Generando {n} muestras ({metodo})...")
            
            # A) Simulación Numérica Continua (Método variable)
            df_new_cont = choose_generation_numeric(metodo, df_cont, n)
            
            # B) Simulación Categórica y Texto (Siempre KNN)
            df_new_disc = simular_datos_knn_categoricos_y_texto(df_disc, n_nuevos=n)
            
            lista_cont.append(df_new_cont)
            lista_disc.append(df_new_disc)
            
    # Concatenar
    df_total_cont = pd.concat(lista_cont, ignore_index=True)
    df_total_disc = pd.concat(lista_disc, ignore_index=True)
    
    dataset_final = pd.concat([df_total_cont, df_total_disc], axis=1)
    print(f"Dataset final generado: {dataset_final.shape}")
    
    return dataset_final

# ==============================================================================
# 6. REGRESIÓN
# ==============================================================================
def variables_regresion(df, variable_objetivo, variables_predictoras):
    """
   Se definen las variables predictoras y la de respuesta
    """
    
    # quedarnos solo con las filas completas
  
    

    X = df[variables_predictoras]
    y = df[variable_objetivo].values.reshape(-1,1)

   
    
    return X, y
def Linear_regression(variable_objetivo, variables_predictoras,data):



    X, y = variables_regresion(data, variable_objetivo, variables_predictoras)




    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size,random_state = 1234)
        
    #agregar comentarios

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
        
    modelo = LinearRegression()
    modelo.fit(X_train_scaled, y_train)
        
    y_pred_train = modelo.predict(X_train_scaled)
    y_pred_test = modelo.predict(X_test_scaled)
        
    # metricas
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # agregar R cuadrada

    print(r2_score(y_test,y_pred_test))

    y_pred = y_pred_train.flatten()
    y_real = y_train.flatten()
    
    # linea de regresión para las predicciones
    z = np.polyfit(y_real, y_pred, 1)
    p = np.poly1d(z)
        
    plt.scatter(y_real, y_pred, alpha=0.6, s=50)
    plt.plot(y_real, p(y_real), "r--", alpha=0.8, linewidth=2, 
                label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
        
    plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 
                'g-', alpha=0.5, linewidth=2, label='descrip perfecta')
        
    plt.xlabel(f'{variable_objetivo} Real ')
    plt.ylabel(f'{variable_objetivo}Predicha')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==============================================================================
# 7. EJECUCIÓN PRINCIPAL
# ==============================================================================

# A. Imputar
df_limpio,categoric_cols = imputar_datos_faltantes(df)

# B. Vectorizar Texto (AHORA SE HACE ANTES DE SIMULAR)
columnas_texto = ['subtitulo','descripcion', 'beneficios', 'aplicaciones']
df_preparado = vectorizar_texto(df_limpio, columnas_texto)

# C. Configuración
cantidades = {
    'knn': 0,        
    'gaussian': 300, 
    'vae': 0       
}

# D. Generación (Simula Categorías y Texto vectorizado con KNN)
df_final_simulado = generar_dataset_con_vectorizacion_simulada(df_preparado, cantidades,categoric_cols)

# E. Regresión
target = 'Carga Timken Ok, lb'
predictors = [
    'Grado NLGI Consistencia',
    'Viscosidad del Aceite Base a 40°C. cSt',
    'Penetración de Cono a 25°C, 0.1mm',
    'Punto de Gota, °C',
    'Estabilidad Mecánica, %',
    'Punto de Soldadura Cuatro Bolas, kgf',
    'Desgaste Cuatro Bolas, mm',
    'Resistencia al Lavado por Agua a 80°C, %',
    'Registro NSF',
    'Factor de Velocidad'
]
print("Ejecutando regresión...")
Linear_regression(target, predictors, df_final_simulado)