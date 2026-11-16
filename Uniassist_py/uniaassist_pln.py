"""AP7 UniAssist: comparação entre regras e TF-IDF + LinearSVC."""
import csv
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PASTA = Path(__file__).resolve().parent
CLASSES = ["Estágio", "Financeiro", "Matrícula", "Suporte"]


def ler_dataset(arquivo):
    with (PASTA / arquivo).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def classificar_regras(texto):
    regras = {
        "Financeiro": ["boleto", "mensalidade", "cobrança", "pagamento", "fatura"],
        "Matrícula": ["matrícula", "rematrícula", "disciplina", "turma", "trancamento"],
        "Estágio": ["estágio", "termo", "supervisor", "concedente", "convênio"],
        "Suporte": ["senha", "login", "portal", "acesso", "aplicativo", "erro"],
    }
    normalizado = texto.casefold()
    pontuacao = {cat: sum(normalizado.count(t.casefold()) for t in termos)
                 for cat, termos in regras.items()}
    maior = max(pontuacao.values())
    vencedores = [cat for cat, pontos in pontuacao.items() if pontos == maior and pontos > 0]
    return vencedores[0] if len(vencedores) == 1 else "Revisão humana"


def exibir_avaliacao(nome, y_true, y_pred):
    print(f"\n=== {nome} ===")
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
    print("Relatório:")
    print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
    print("Matriz de confusão (linhas = real; colunas = previsto)")
    print("Ordem:", CLASSES)
    print(confusion_matrix(y_true, y_pred, labels=CLASSES))


def main():
    registros = ler_dataset("mensagens.csv")
    textos = [r["texto"] for r in registros]
    rotulos = [r["categoria"] for r in registros]
    print("Quantidade de exemplos:", len(registros))
    print("Distribuição:", dict(Counter(rotulos)))

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        textos, rotulos, test_size=0.25, random_state=42, stratify=rotulos
    )
    modelo = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("svm", LinearSVC(random_state=42)),
    ])
    modelo.fit(X_treino, y_treino)
    pred_ml = modelo.predict(X_teste)
    exibir_avaliacao("TF-IDF + LinearSVC (unigramas e bigramas)", y_teste, pred_ml)

    pred_regras = [classificar_regras(t) for t in X_teste]
    acuracia_regras = sum(p == y for p, y in zip(pred_regras, y_teste)) / len(y_teste)
    print("\n=== Regras ===")
    print(f"Acurácia automatizada: {acuracia_regras:.4f}")
    labels_regras = CLASSES + ["Revisão humana"]
    print("Matriz de confusão das regras; ordem:", labels_regras)
    print(confusion_matrix(y_teste, pred_regras, labels=labels_regras))

    modelo_uni = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), lowercase=True)),
        ("svm", LinearSVC(random_state=42)),
    ])
    modelo_uni.fit(X_treino, y_treino)
    pred_uni = modelo_uni.predict(X_teste)
    print("\n=== Experimento: somente unigramas ===")
    print(f"Acurácia: {accuracy_score(y_teste, pred_uni):.4f}")

    print("\n=== Oito mensagens inéditas ===")
    for item in ler_dataset("desafios.csv"):
        texto = item["texto"]
        print(f'{item["id"]}: {texto}')
        print("  Regras:", classificar_regras(texto))
        print("  TF-IDF + LinearSVC:", modelo.predict([texto])[0])


if __name__ == "__main__":
    main()
