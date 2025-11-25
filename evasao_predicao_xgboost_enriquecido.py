"""
==============================================================================
PREDIÇÃO DE EVASÃO ACADÊMICA - XGBOOST COM DADOS ENRIQUECIDOS
==============================================================================
Autor: Vitória de Lourdes Carvalho Santos
Dataset: financeiro_enriquecido.csv (com dados de accounts.csv)
Objetivo: Modelo XGBoost com features adicionais
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import xgboost as xgb
import warnings
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

print("🎯 PREDIÇÃO DE EVASÃO - XGBOOST COM DADOS ENRIQUECIDOS")
print("=" * 70)

# ==============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================

def carregar_dados_enriquecidos(arquivo='financeiro_enriquecido.csv'):
    """Carrega e prepara os dados enriquecidos"""
    print(f"\n📊 Carregando dados enriquecidos do {arquivo}...")
    
    df = pd.read_csv(arquivo, low_memory=False)
    print(f"Dataset carregado: {len(df):,} registros e {len(df.columns)} colunas")
    
    # Criar variável target
    df['target_evadido'] = df['ind_evadido'].apply(lambda x: 1 if x == 'Sim' else 0)
    
    # Verificar distribuição
    print(f"\nDistribuição da variável target:")
    nao_evadidos = (df['target_evadido'] == 0).sum()
    evadidos = (df['target_evadido'] == 1).sum()
    print(f"Não evadidos: {nao_evadidos:,} ({nao_evadidos/len(df)*100:.1f}%)")
    print(f"Evadidos: {evadidos:,} ({evadidos/len(df)*100:.1f}%)")
    
    return df

def extrair_modalidade(nome_curso):
    """Extrai modalidade do nome do curso"""
    if pd.isna(nome_curso):
        return 'Desconhecido'
    nome = str(nome_curso).upper()
    if 'EAD' in nome or 'DISTÂNCIA' in nome:
        return 'EAD'
    elif 'ONLINE' in nome:
        return 'Online'
    elif 'HÍBRIDO' in nome:
        return 'Híbrido'
    else:
        return 'Presencial'

def extrair_campus(nome_curso):
    """Extrai unidade/campus do nome do curso"""
    if pd.isna(nome_curso):
        return 'Desconhecido'
    nome = str(nome_curso).upper()
    # Padrões comuns: PPL, PSG, PBE, PBR, PMG, etc
    match = re.search(r'- ([A-Z]{3,4})', nome)
    if match:
        return match.group(1)
    return 'Outro'

def calcular_dias_desde_criacao(created_at):
    """Calcula dias desde a criação do curso"""
    if pd.isna(created_at):
        return None
    try:
        data = pd.to_datetime(created_at)
        hoje = datetime.now()
        return (hoje - data).days
    except:
        return None

def criar_features_enriquecidas(df):
    """Cria features incluindo as novas do accounts"""
    print("\n🔧 Criando features enriquecidas para o modelo...")
    
    # Features numéricas originais
    features_numericas = [
        'idade_aluno',
        'qtd_semestres_cursados',
        'media_frequencia_anterior',
        'media_nota_anterior',
        'val_distancia_campus',
        'qtd_reprovacoes_curso',
        'qtd_disc_reprov_nota_curso',
        'qtd_disc_reprov_frequencia_curso',
        'perc_cursado'
    ]
    
    # Features categóricas originais
    features_categoricas = [
        'sexo_aluno',
        'dsc_turno',
        'dsc_forma_ingresso',
        'ind_possui_bolsa',
        'ind_inadimplente',
        'ind_possui_financiamento'
    ]
    
    # Criar dataset de features
    df_features = df[features_numericas + features_categoricas + ['target_evadido']].copy()
    
    # NOVAS FEATURES DO ACCOUNTS
    print("\n🆕 Criando features do accounts.csv...")
    
    # 1. Modalidade do curso (EAD, Online, Presencial)
    if 'name' in df.columns:
        df_features['modalidade_curso'] = df['name'].apply(extrair_modalidade)
        print(f"  ✓ Modalidade extraída: {df_features['modalidade_curso'].value_counts().to_dict()}")
    
    # 2. Campus/Unidade
    if 'name' in df.columns:
        df_features['campus'] = df['name'].apply(extrair_campus)
        print(f"  ✓ Campus extraído: Top 5 = {df_features['campus'].value_counts().head().to_dict()}")
    
    # 3. Dias desde criação do curso (maturidade)
    if 'created_at' in df.columns:
        df_features['dias_desde_criacao'] = df['created_at'].apply(calcular_dias_desde_criacao)
        df_features['dias_desde_criacao'].fillna(df_features['dias_desde_criacao'].median(), inplace=True)
        print(f"  ✓ Dias desde criação calculados (média: {df_features['dias_desde_criacao'].mean():.0f} dias)")
    
    # 4. Workflow state (active/deleted)
    if 'workflow_state' in df.columns:
        df_features['curso_ativo'] = df['workflow_state'].apply(lambda x: 1 if x == 'active' else 0)
        print(f"  ✓ Estado do curso: {df_features['curso_ativo'].value_counts().to_dict()}")
    
    # Preencher valores ausentes das features originais
    for col in features_numericas:
        if col in df_features.columns:
            df_features[col].fillna(df_features[col].median(), inplace=True)
    
    for col in features_categoricas:
        if col in df_features.columns:
            df_features[col].fillna(df_features[col].mode()[0], inplace=True)
    
    # Criar features derivadas originais
    df_features['performance_combinada'] = (
        df_features['media_nota_anterior'] + df_features['media_frequencia_anterior']
    ) / 2
    
    df_features['taxa_reprovacao'] = (
        df_features['qtd_reprovacoes_curso'] / 
        (df_features['qtd_semestres_cursados'] + 1)
    )
    
    # Criar faixa etária
    df_features['faixa_etaria'] = pd.cut(
        df_features['idade_aluno'],
        bins=[0, 20, 25, 30, 100],
        labels=['Até 20', '21-25', '26-30', '31+']
    )
    
    # Adicionar novas features categóricas à lista
    novas_categoricas = []
    if 'modalidade_curso' in df_features.columns:
        novas_categoricas.append('modalidade_curso')
    if 'campus' in df_features.columns:
        novas_categoricas.append('campus')
    
    features_categoricas_completas = features_categoricas + novas_categoricas + ['faixa_etaria']
    
    # Label Encoding para variáveis categóricas
    label_encoders = {}
    for col in features_categoricas_completas:
        if col in df_features.columns:
            le = LabelEncoder()
            df_features[f'{col}_encoded'] = le.fit_transform(df_features[col].astype(str))
            label_encoders[col] = le
    
    # Selecionar features finais
    features_finais = (
        features_numericas + 
        ['performance_combinada', 'taxa_reprovacao'] +
        [f'{col}_encoded' for col in features_categoricas_completas]
    )
    
    # Adicionar features numéricas novas
    if 'dias_desde_criacao' in df_features.columns:
        features_finais.append('dias_desde_criacao')
    if 'curso_ativo' in df_features.columns:
        features_finais.append('curso_ativo')
    
    # Remover features originais categóricas
    df_features = df_features[features_finais + ['target_evadido']]
    
    print(f"\n✅ Total de features: {len(features_finais)}")
    for i, feat in enumerate(features_finais, 1):
        print(f"  {i:2d}. {feat}")
    
    return df_features, features_finais, label_encoders

# ==============================================================================
# 2. TREINAMENTO DO MODELO XGBOOST
# ==============================================================================

def treinar_modelo_xgboost_otimizado(df_features, features_finais, amostra_size=15000):
    """Treina modelo XGBoost otimizado com dados enriquecidos"""
    print(f"\n🤖 Treinando modelo XGBoost OTIMIZADO (com dados enriquecidos)...")
    
    # Usar amostra maior por ter mais features
    if len(df_features) > amostra_size:
        print(f"Dataset grande ({len(df_features):,} registros). Usando amostra de {amostra_size:,} registros.")
        _, df_sample = train_test_split(
            df_features, 
            test_size=amostra_size/len(df_features), 
            random_state=42, 
            stratify=df_features['target_evadido']
        )
        df_features = df_sample
    
    # Separar features e target
    X = df_features[features_finais]
    y = df_features['target_evadido']
    
    print(f"Shape dos dados: X={X.shape}, y={y.shape}")
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Treino: {len(X_train):,} registros")
    print(f"Teste: {len(X_test):,} registros")
    
    # Calcular scale_pos_weight para balanceamento
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight (balanceamento): {scale_pos_weight:.2f}")
    
    print("\nTreinando modelo XGBoost com hiperparâmetros otimizados...")
    
    modelo = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    
    modelo.fit(X_train, y_train, verbose=False)
    
    # Fazer predições
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    
    print(f"✅ Modelo treinado com {modelo.n_estimators} árvores")
    
    return modelo, X_test, y_test, y_pred, y_pred_proba, features_finais

# ==============================================================================
# 3. AVALIAÇÃO DO MODELO
# ==============================================================================

def avaliar_modelo(modelo, X_test, y_test, y_pred, y_pred_proba):
    """Avalia o desempenho do modelo"""
    print(f"\n📊 RESULTADOS DO MODELO ENRIQUECIDO:")
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print("\nRelatório de classificação:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Não Evadido', 'Evadido'],
        digits=2
    ))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusão:")
    print(cm)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC-ROC: {auc:.3f} ({auc*100:.1f}%)")
    
    return accuracy, cm, auc

# ==============================================================================
# 4. VISUALIZAÇÕES
# ==============================================================================

def plotar_importancia_features(modelo, features_finais):
    """Plota importância das features"""
    print("\n🔍 ANÁLISE DE IMPORTÂNCIA DAS FEATURES:")
    
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    df_importance = pd.DataFrame({
        'feature': [features_finais[i] for i in indices],
        'importance': importances[indices]
    })
    
    print("\nTop 15 features mais importantes:")
    for i, row in df_importance.head(15).iterrows():
        print(f"  {i+1:2d}. {row['feature']:45s} {row['importance']:.3f} ({row['importance']*100:.1f}%)")
    
    # Plotar
    plt.figure(figsize=(14, 10))
    
    top_n = 20
    df_plot = df_importance.head(top_n)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_plot)))
    
    plt.barh(range(len(df_plot)), df_plot['importance'], color=colors)
    plt.yticks(range(len(df_plot)), df_plot['feature'])
    plt.xlabel('Importância', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('Top 20 Features Mais Importantes - XGBoost Enriquecido\nPredição de Evasão Acadêmica', 
              fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        plt.text(row['importance'], i, f" {row['importance']:.3f}", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('Importancia_Features_XGBoost_Enriquecido.png', dpi=300, bbox_inches='tight')
    print("\n📸 Gráfico salvo: Importancia_Features_XGBoost_Enriquecido.png")
    plt.close()
    
    df_importance.to_csv('importancia_features_xgboost_enriquecido.csv', index=False)
    print("💾 Dados salvos: importancia_features_xgboost_enriquecido.csv")
    
    return df_importance

def plotar_matriz_confusao(cm):
    """Plota matriz de confusão"""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Não Evadido', 'Evadido'],
                yticklabels=['Não Evadido', 'Evadido'],
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    plt.title('Matriz de Confusão - XGBoost Enriquecido\nPredição de Evasão Acadêmica', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Valor Real', fontsize=12, fontweight='bold')
    plt.xlabel('Valor Predito', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Matriz_Confusao_XGBoost_Enriquecido.png', dpi=300, bbox_inches='tight')
    print("📸 Gráfico salvo: Matriz_Confusao_XGBoost_Enriquecido.png")
    plt.close()

def plotar_curva_roc(y_test, y_pred_proba, auc):
    """Plota curva ROC"""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12, fontweight='bold')
    plt.title('Curva ROC - XGBoost Enriquecido\nPredição de Evasão Acadêmica', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Curva_ROC_XGBoost_Enriquecido.png', dpi=300, bbox_inches='tight')
    print("📸 Gráfico salvo: Curva_ROC_XGBoost_Enriquecido.png")
    plt.close()

# ==============================================================================
# 5. EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    """Função principal"""
    try:
        # Carregar dados enriquecidos
        df = carregar_dados_enriquecidos()
        
        # Criar features enriquecidas
        df_features, features_finais, label_encoders = criar_features_enriquecidas(df)
        
        # Treinar modelo
        modelo, X_test, y_test, y_pred, y_pred_proba, features_finais = treinar_modelo_xgboost_otimizado(
            df_features, features_finais
        )
        
        # Avaliar modelo
        accuracy, cm, auc = avaliar_modelo(modelo, X_test, y_test, y_pred, y_pred_proba)
        
        # Análise de importância
        df_importance = plotar_importancia_features(modelo, features_finais)
        
        # Visualizações
        print("\n📊 Criando visualizações...")
        plotar_matriz_confusao(cm)
        plotar_curva_roc(y_test, y_pred_proba, auc)
        
        print("\n" + "=" * 70)
        print("🎯 ANÁLISE CONCLUÍDA!")
        print("=" * 70)
        print(f"✅ Modelo enriquecido treinado com sucesso")
        print(f"✅ Acurácia: {accuracy*100:.1f}%")
        print(f"✅ AUC-ROC: {auc*100:.1f}%")
        print(f"✅ Features totais: {len(features_finais)} (incluindo novas do accounts)")
        
        print("\n🔝 TOP 10 FEATURES MAIS IMPORTANTES:")
        for i, row in df_importance.head(10).iterrows():
            emoji = "🆕" if any(x in row['feature'] for x in ['modalidade', 'campus', 'dias_desde', 'curso_ativo']) else "  "
            print(f"  {emoji} {i+1:2d}. {row['feature']}: {row['importance']*100:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
