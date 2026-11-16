# Parecer técnico — AP7: Classificação de mensagens da UniAssist

## Resultados no conjunto de teste

A base possui 80 mensagens sintéticas, balanceadas: 20 por categoria. A divisão estratificada, com `test_size=0.25` e `random_state=42`, gerou 60 exemplos de treino e 20 de teste (5 por classe).

| Método | Acurácia no teste | Observação |
|---|---:|---|
| Regras por palavras-chave | **0,8000 (80%)** | 16 acertos em 20; encaminhamentos para revisão humana contam como não acerto automatizado. |
| TF-IDF + LinearSVC (1–2 gramas) | **0,7000 (70%)** | 14 acertos em 20. |

Neste conjunto e nesta divisão, o baseline de regras teve desempenho 10 pontos percentuais superior ao modelo. Isso não prova que regras sejam melhores em geral: a base é pequena, sintética e contém vocabulário muito próximo das palavras-chave escolhidas.

## Relatório de classificação — TF-IDF + LinearSVC

| Classe | Precisão | Revocação (recall) | F1-score | Suporte |
|---|---:|---:|---:|---:|
| Estágio | 0,71 | 1,00 | 0,83 | 5 |
| Financeiro | 0,50 | 0,40 | 0,44 | 5 |
| Matrícula | 0,75 | 0,60 | 0,67 | 5 |
| Suporte | 0,80 | 0,80 | 0,80 | 5 |
| **Acurácia** | | | **0,70** | **20** |
| Macro avg | 0,69 | 0,70 | 0,69 | 20 |
| Weighted avg | 0,69 | 0,70 | 0,69 | 20 |

## Matrizes de confusão

Para o modelo, linhas representam a classe real e colunas a classe prevista. Ordem das classes: **Estágio, Financeiro, Matrícula, Suporte**.

```text
[[5 0 0 0]
 [1 2 1 1]
 [0 2 3 0]
 [1 0 0 4]]
```

O modelo acertou todos os cinco exemplos de Estágio. A maior dificuldade foi Financeiro: acertou 2 de 5, confundindo os demais com Estágio, Matrícula e Suporte. Matrícula teve 3 acertos e 2 previsões como Financeiro; Suporte teve 4 acertos e 1 previsão como Estágio.

Para as regras, foi incluída a saída adicional **Revisão humana**. Ordem das colunas: **Estágio, Financeiro, Matrícula, Suporte, Revisão humana**. As linhas seguem as quatro classes reais na ordem anterior.

```text
[[4 0 0 0 1]
 [0 5 0 0 0]
 [0 0 4 0 1]
 [0 0 0 3 2]]
```

As regras acertaram 16 mensagens e encaminharam 4 para revisão humana. A linha de Revisão humana não aparece porque o conjunto de teste não possui exemplos cuja categoria real seja essa saída.

## Mensagens inéditas e ambíguas

As mensagens de desafio não possuem rótulos oficiais. Portanto, as observações abaixo são análise semântica, não medição de acerto.

| ID | Regras | Modelo | O que dificultou? | Política proposta |
|---|---|---|---|---|
| C01 | Revisão humana | Suporte | Mistura acesso/senha (Suporte) com boleto (Financeiro). | Revisar e perguntar se o problema principal é emitir/acessar o boleto ou recuperar a senha. |
| C02 | Revisão humana | Financeiro | “Quitar” e “parcela” não estão nas palavras-chave; o modelo generaliza pelo contexto. | Incluir sinônimos validados nas regras; confirmar se a parcela é cobrança institucional. |
| C05 | Estágio | Estágio | A mensagem combina falha do sistema (Suporte) com relatório de estágio (Estágio). | Encaminhar ao setor responsável pelo relatório e abrir apoio técnico se a falha impedir o envio. |
| C06 | Matrícula | Matrícula | Há pagamento de taxa (Financeiro) e rematrícula pendente (Matrícula). | Tratar como caso intersetorial: Financeiro verifica compensação e Matrícula verifica a efetivação. |
| C08 | Revisão humana | Matrícula | “Registrar horas” sugere Estágio, mas não há termo explícito do dicionário; o modelo escolhe Matrícula. | Encaminhar para triagem humana/Estágio, confirmando se são horas de estágio curricular. |

Outros resultados: C03 foi classificada como Matrícula pelos dois métodos; C04 como Estágio pelos dois; C07 como Suporte pelos dois. Em C01, C02 e C08, a divergência demonstra que a classificação automática não deve ser tratada como decisão definitiva.

## Experimento com unigramas

Ao trocar `ngram_range=(1, 2)` por `ngram_range=(1, 1)`, mantendo a mesma divisão, o modelo obteve **0,7000 (70%)**, igual ao resultado com unigramas e bigramas. Nesta amostra, os bigramas não alteraram a acurácia. Como o mesmo conjunto de teste foi consultado em mais de uma configuração, essa comparação é exploratória: escolher repetidamente parâmetros com base no teste pode adaptar indiretamente a solução a ele. Uma avaliação mais robusta exigiria validação cruzada no treino e um conjunto de teste final separado.

## Parecer final (até 15 linhas)

O baseline por regras alcançou 80% de acurácia, enquanto TF-IDF + LinearSVC alcançou 70% nesta divisão.  
As regras foram eficazes quando as mensagens continham termos previstos, mas encaminharam casos sem correspondência ou com empate para revisão.  
O modelo conseguiu reconhecer alguns sinônimos e expressões não previstas, porém confundiu mensagens de Financeiro com outras categorias.  
A base possui somente 80 exemplos sintéticos, o que limita a generalização para mensagens reais e variadas.  
As mensagens podem tratar de mais de um setor, algo que um classificador de rótulo único não representa adequadamente.  
Recomenda-se usar a automação como triagem inicial, nunca como decisão final infalível.  
Casos sem confiança, com sinais de múltiplos setores ou com impacto financeiro/acadêmico devem ir para revisão humana.  
Quando houver duas demandas, o atendimento deve registrar ambas e encaminhar cada parte ao setor competente.  
As palavras-chave e o modelo devem ser revisados periodicamente com exemplos reais anonimizados e rotulados por pessoas.  
Antes da implantação, é necessário ampliar a base, avaliar novos dados e monitorar erros por categoria.
