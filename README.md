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
- `--seed`: controla a geração pseudo-aleatória para reproduzir o cenário.

### Camadas do mapa

O HTML final inclui duas camadas em um `LayerControl` do próprio Folium:

- **Pontos (sem rotas)**: mostra apenas depósitos e entregas para inspeção rápida.
- **Rotas e pontos**: mesma camada de pontos, acrescida dos caminhos otimizados por caminhão.

Use o seletor no canto do mapa para alternar entre versões sem precisar rodar o script novamente.

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
