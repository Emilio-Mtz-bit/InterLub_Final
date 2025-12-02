
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Simulation_and_Regression_Functions import vectorizar_texto

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================

@st.cache_data
def load_data():
    """
    Carga y procesa los datos.
    """
    try:
        
        df = pd.read_csv('../data/dataset_simulado.csv')
    except FileNotFoundError:
        st.error("El archivo `../data/datos_onehotenocder.csv` no fue encontrado. Por favor, asegúrese de que el archivo existe en el directorio `data`.")
        return None, None

    df_recommender = pd.read_csv('../data/datos_onehotenocder.csv')
    columnas_texto = ['subtitulo','descripcion', 'beneficios', 'aplicaciones']
    df_recommender = vectorizar_texto(df_recommender,columnas_texto)
    
    # Asegurarse de que las columnas de texto originales sean eliminadas si vectorizar_texto no lo hizo
    df_recommender = df_recommender.drop(columns=columnas_texto, errors='ignore')
    
    return df, df_recommender,columnas_texto

data, recommender_data,columnas_texto = load_data()

if data is not None:
    st.title("Dashboard InterLub")
    st.markdown("### Un dashboard para explorar y recomendar grasas lubricantes.")

    tab1, tab2 = st.tabs(["Recomendador de Grasas", "Análisis de Regresión"])

    # ==============================================================================
    # TAB 1: RECOMENDADOR DE GRASAS
    # ==============================================================================
    with tab1:
        st.header("Encuentre la grasa ideal")
        st.markdown("Seleccione las características deseadas para encontrar las 5 grasas más similares.")
        grease_description = st.text_area("Describa la grasa que necesita:")


        if recommender_data is not None:
            # --- Separar columnas por tipo para una mejor UI ---
            all_numeric_cols = recommender_data.select_dtypes(include=np.number).columns.tolist()
            
            binary_cols = []
            continuous_cols = []

            for col in all_numeric_cols:
                if col.startswith(tuple(columnas_texto)):
                     continue
                if set(recommender_data[col].unique()).issubset({0, 1}):
                    binary_cols.append(col)
              
                # Esto también excluye las columnas de texto vectorizadas (TF-IDF) que suelen estar entre 0 y 1.
                elif recommender_data[col].max() > 1:
                    continuous_cols.append(col)
            
            selectable_features = continuous_cols + binary_cols
            
            selected_features = st.multiselect("Seleccione las variables para el vector de búsqueda:", options=selectable_features)
            
            input_vector_sliders = {}
            for feature in selected_features:
                if feature in binary_cols:
                    input_vector_sliders[feature] = st.selectbox(f"Valor para {feature}", options=[0, 1])
                else: # Continuous cols
                    min_val = float(recommender_data[feature].min())
                    max_val = float(recommender_data[feature].max())
                    mean_val = float(recommender_data[feature].mean())
                    input_vector_sliders[feature] = st.slider(f"Valor para {feature}", min_value=min_val, max_value=max_val, value=mean_val)

            if st.button("Buscar grasas similares"):
                # Validar que haya algún input
                if not grease_description.strip() and not selected_features:
                    st.warning("Por favor, describa la grasa o seleccione al menos un parámetro técnico.")
                else:
                    # Crear y vectorizar el input del usuario solo al hacer click
                    if grease_description.strip():
                        input_df = pd.DataFrame([{"text": grease_description}])
                        input_vectorized_df = vectorizar_texto(input_df, ["text"])
                    else:
                        input_vectorized_df = pd.DataFrame()

                    # Llenar el vector de búsqueda
                    search_vector = pd.Series(0.0, index=recommender_data.columns)

                    # Transferir valores de sliders
                    for feature, value in input_vector_sliders.items():
                        if feature in search_vector.index:
                            search_vector[feature] = value
                    
                    # Transferir valores de texto vectorizado
                    if not input_vectorized_df.empty:
                        for col in input_vectorized_df.columns:
                            if col in search_vector.index:
                                search_vector[col] = input_vectorized_df[col].iloc[0]

                    # Asegurar que no haya NaNs que puedan causar problemas
                    search_vector.fillna(0, inplace=True)

                    # --- Preparación para la Similaridad ---
                    # Seleccionar solo columnas numéricas para el cálculo
                    recommender_data_numeric = recommender_data.select_dtypes(include=np.number)
                    
                    # Alinear el search_vector con las columnas numéricas
                    search_vector_numeric = search_vector[recommender_data_numeric.columns]

                    # Convertir a arrays de numpy
                    X_data_array = np.nan_to_num(recommender_data_numeric.to_numpy())
                    vector_2d = np.nan_to_num(search_vector_numeric.to_numpy().reshape(1, -1))
                    
                    # Validar dimensiones antes de la similaridad
                    if X_data_array.shape[1] != vector_2d.shape[1]:
                        st.error(f"Error de dimensiones: El vector de datos tiene {X_data_array.shape[1]} características y el vector de búsqueda tiene {vector_2d.shape[1]}. No se puede calcular la similaridad.")
                    else:
                                            simil = cosine_similarity(X_data_array, vector_2d)
                                            
                                            idx = np.argsort(simil.flatten())[::-1]
                                            
                                            top_5_idx = idx[:5]
                                            top_5_scores = simil.flatten()[top_5_idx]
                        
                                            st.markdown("#### Top 5 Grasas Recomendadas")
                                            
                                            results_df = pd.DataFrame({
                                                "Código de Grasa": recommender_data['codigoGrasa'].iloc[top_5_idx],
                                                "Similaridad": top_5_scores
                                            })
                                            st.dataframe(results_df)
                                            
                                                # ==============================================================================
    # TAB 2: ANÁLISIS DE REGRESIÓN
    # ==============================================================================
    with tab2:
        st.header("Análisis de Regresión Lineal")
        st.markdown("Explore la relación entre variables y prediga valores.")

        if data is not None:
            # Filtrar columnas 'Unnamed'
            all_numerical_cols_reg = data.select_dtypes(include=np.number).columns.tolist()
            filter_tuple =tuple(['Unnamed'] + columnas_texto)
            numerical_cols_reg = [col for col in all_numerical_cols_reg if not col.startswith(filter_tuple) ]
            
            # Evitar que la variable objetivo esté en las predictoras
            target_variable = st.selectbox("Seleccione la variable objetivo (Y):", options=numerical_cols_reg)
            
            available_predictors = [col for col in numerical_cols_reg if col != target_variable]
            
            predictor_variables = st.multiselect("Seleccione las variables predictoras (X):", options=available_predictors)

            if st.button("Ejecutar Regresión"):
                if not predictor_variables:
                    st.warning("Por favor, seleccione al menos una variable predictora.")
                else:
                    try:
                        # Preparar datos
                        df_reg = data[[target_variable] + predictor_variables].dropna()
                        X = df_reg[predictor_variables]
                        y = df_reg[target_variable]

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        model = LinearRegression()
                        model.fit(X_train_scaled, y_train)

                        y_pred = model.predict(X_test_scaled)

                        # Métricas
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        st.markdown("#### Resultados de la Regresión")
                        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
                        st.metric(label="R-squared (R²)", value=f"{r2:.4f}")

                        # Coeficientes
                        st.markdown("#### Coeficientes del Modelo")
                        coef_df = pd.DataFrame(model.coef_, index=predictor_variables, columns=["Coeficiente"])
                        st.dataframe(coef_df)

                        # Gráfico de Predicciones vs. Valores Reales
                        st.markdown("#### Gráfico de Predicciones vs. Valores Reales")
                        fig, ax = plt.subplots()
                        ax.scatter(y_test, y_pred, alpha=0.6)
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                        ax.set_xlabel("Valores Reales")
                        ax.set_ylabel("Predicciones")
                        ax.set_title("Predicciones vs. Valores Reales")
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Ocurrió un error al ejecutar la regresión: {e}")

