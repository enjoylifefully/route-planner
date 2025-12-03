# Planejamento de rotas com K-Means e 2-opt

Projeto em Python que cria um cenário simples de entregas em São Paulo com
três caminhões. Os passos implementados são:

- download da malha viária via **OSMnx**;
- divisão dos pontos de entrega usando **K-Means** do scikit-learn;
- otimização da ordem das entregas em cada cluster com **2-opt** (lib
  `python-tsp`);
- renderização do resultado em um mapa **Folium** contendo depósitos, entregas
  e rotas individuais.

## Rodando o projeto

```bash
uv run python -m route_planner
```

O script salva `output/map.html` com todas as camadas. Abra o arquivo no
navegador para visualizar o mapa e verifique o terminal para um resumo textual
da ordem de visitas de cada caminhão.

### Escolhendo quantidades

Parâmetros importantes podem ser ajustados pela linha de comando:

```bash
uv run python -m route_planner --num-trucks 4 --num-deliveries 20 --radius-m 15000 --seed 123
```

- `--num-trucks`: quantidade de depósitos/caminhões (`k`).
- `--num-deliveries`: número total de entregas (`n`, deve ser ≥ `k`).
- `--radius-m`: raio usado tanto para gerar os pontos quanto para baixar o grafo via OSMnx.
- `--seed`: controla a geração pseudo-aleatória; se não for informado, o script
  escolhe uma semente aleatória automaticamente e informa no terminal.

### Camadas do mapa

O HTML final inclui várias camadas no `LayerControl` para enxergar cada etapa dos
algoritmos:

- **Pontos (sem rotas)**: depósitos e entregas em cinza.
- **Clusters - distribuição inicial**: ligações (tracejadas) entre caminhões e
  entregas antes de aplicar K-Means.
- **Clusters - K-Means**: mesmas ligações, agora usando os grupos calculados.
- **Rotas - ordem sequencial**: caminho percorrido caso o caminhão siga a
  sequência original das entregas (pré 2-opt).
- **Rotas - 2-opt**: resultado otimizando a ordem das visitas em cada grupo.

Ative/desative as camadas desejadas para comparar visualmente as melhorias em
cada componente.

### Comparando melhorias numéricas

Além do mapa, o terminal imprime dois blocos com métricas:

1. **Agrupamento** – distância total (linha reta) caminhão→entregas antes e
   depois do K-Means.
2. **Rotas** – comprimento real de cada rota na ordem original versus depois do
   2-opt (em quilômetros e porcentagem de ganho), junto com a sequência final
   das visitas.

## Personalização

- Ajuste as funções `generate_trucks`/`generate_deliveries` em
  `route_planner/__main__.py` para
  carregar dados reais (de um CSV, banco ou API) em vez de gerar amostras
  pseudo-aleatórias.
- Use o parâmetro `--radius-m` ou edite a função `build_graph` se precisar de um
  recorte maior/menor do grafo do OSM.
- Novos caminhões ou entregas são tratados automaticamente pelo K-Means (desde
  que `n` ≥ `k`).
- Os algoritmos estão modularizados: `clustering.py` concentra o K-Means e
  `route_optimizer.py` a etapa 2-opt, o que facilita substituições por outras
  abordagens caso necessário.
- Todo o código da aplicação fica dentro da pasta `route_planner/`, deixando a
  raiz do repositório reservada para arquivos auxiliares (README, configs, etc.).
