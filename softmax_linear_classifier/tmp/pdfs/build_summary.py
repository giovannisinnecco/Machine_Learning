from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

ROOT=Path.cwd(); OUT=ROOT/'output/pdf/resumo_decisoes_modelos.pdf'
M={k:json.loads((ROOT/f'results/{stem}_metricas.json').read_text()) for k,stem in [('1a','softmax_1a'),('1b','knn_1b'),('2a','softmax_2a'),('2b','knn_2b')]}
B={k:json.loads((ROOT/f'results/{stem}_busca.json').read_text()) for k,stem in [('1a','softmax_1a'),('1b','knn_1b'),('2a','softmax_2a'),('2b','knn_2b')]}
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=colors.HexColor('#14344B'),spaceAfter=16))
styles.add(ParagraphStyle(name='SubX',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=colors.HexColor('#137D8D'),spaceBefore=10,spaceAfter=6))
styles.add(ParagraphStyle(name='BodyX',fontName='Helvetica',fontSize=9.4,leading=13,spaceAfter=6))
styles.add(ParagraphStyle(name='SmallX',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=4))
styles.add(ParagraphStyle(name='CellX',fontName='Helvetica',fontSize=8.1,leading=10))
story=[]
def p(t,style='BodyX'): story.append(Paragraph(t,styles[style]))
def h(t):p(t,'SubX')
def title(t,sub):p(t,'TitleX');p(sub,'SmallX')
def table(rows,widths=None,small=False):
 rows=[[Paragraph(escape(str(x)),styles['CellX']) for x in row] for row in rows]
 t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E0EDF1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,0),(-1,0),0.6,colors.HexColor('#137D8D')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F8FA')])]))
 story.append(t);story.append(Spacer(1,7))
def page():story.append(PageBreak())
def pct(v):return f'{v*100:.2f}%'.replace('.',',')
def num(v):return f'{v:.6f}'.replace('.',',')
def footer(c,d):
 c.setStrokeColor(colors.HexColor('#CBD8DF'));c.line(40,39,555,39)
 c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#587080'))
 c.drawString(40,27,'IA048 | Atividade 2 | Registro das decisões e resultados')
 c.drawRightString(555,27,f'{d.page}')

title('Geral | Protocolo experimental','IA048 - Atividade 2 • Registro da implementação atual • 27/09/2026')
p('Este documento reúne as decisões efetivamente implementadas nos quatro notebooks, seus hiperparâmetros, procedimentos de seleção, ajustes finais e resultados. As tabelas são geradas a partir dos arquivos JSON salvos, sem executar uma nova busca. A nomenclatura 2a/2b corresponde à repetição dos métodos da primeira parte com sinais temporais, solicitada na segunda parte do enunciado.')
h('Dados, classes e representações')
p('Base UCI HAR: 10.299 janelas de 30 participantes, com seis classes. Rótulos: <b>1</b> caminhar; <b>2</b> subir escadas; <b>3</b> descer escadas; <b>4</b> sentado; <b>5</b> em pé; <b>6</b> deitado. Cada janela contém 128 instantes a 50 Hz (2,56 s).')
table([['Itens','Entrada','Classificador'],['1a / 1b','561 atributos extraídos fornecidos pela base','Softmax / KNN'],['2a / 2b','768 valores: 6 sinais concatenados x 128 instantes','Softmax / KNN']],[65,290,160])
p('Nos itens 2a/2b, a ordem dos blocos é <b>total_acc_x, total_acc_y, total_acc_z, body_gyro_x, body_gyro_y, body_gyro_z</b>, preservando a ordem dos 128 instantes de cada arquivo em Inertial Signals. Usamos total_acc para ACC, incluindo a componente gravitacional; não usamos body_acc. “Brutos” identifica os sinais disponibilizados pela base; não implica sinais sem o pré-processamento de origem. Não se aplicam novos filtros, extração de características, PCA ou seleção de atributos.')
h('Divisão oficial e separação por sujeito')
table([['Etapa','Participantes','Amostras'],['Treino oficial / desenvolvimento','21 (70% dos participantes)','7.352'],['Teste oficial, comum aos quatro modelos','9 (30% dos participantes)','2.947'],['Treino interno do Softmax','17, dentro do treino oficial','5.800'],['Validação interna do Softmax','4, dentro do treino oficial','1.552']],[285,145,85])
p('<b>Desenvolvimento:</b> 1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30.<br/><b>Teste:</b> 2, 4, 9, 10, 12, 13, 18, 20, 24.<br/><b>Validação fixa do Softmax:</b> 17, 25, 26, 30; o treino interno contém os outros 17 sujeitos do desenvolvimento.')
p('A validação fixa usa np.random.default_rng(42).permutation sobre os IDs ordenados do treino oficial; os quatro primeiros sorteados formam a validação. O KNN reúne novamente treino e validação para formar os cinco folds por sujeito. Nenhuma pessoa pertence simultaneamente ao desenvolvimento e ao teste. Todas as amostras disponíveis são mantidas; não há reamostragem de classes.')
p('A divisão experimental inicial de 20/4/6 sujeitos foi substituída pela divisão oficial acima. Os resultados atuais e este documento usam exclusivamente a versão corrigida. O teste não participa da escolha dos hiperparâmetros nem do ajuste dos modelos.')
page()
title('Geral | Preparação e avaliação','Continuação das decisões comuns e diferenças de protocolo')
h('Padronização e verificações dos dados')
p('Padronização por atributo: z = (x - média) / desvio-padrão. Usa-se desvio populacional (ddof=0); se o desvio é zero, a escala é substituída por 1. Cálculos dos modelos em float64. Não há imputação: valores não finitos, dimensões incompatíveis ou classes inesperadas provocam erro.')
p('<b>Softmax (1a/2a):</b> prepare_data(..., standardize=True) estima média e escala somente nas 5.800 amostras de treino interno. Os mesmos valores transformam validação e teste. No ajuste final sobre 7.352 amostras, esses parâmetros são mantidos, sem recalcular a padronização.<br/><b>KNN (1b/2b):</b> prepare_data(..., standardize=False) entrega dados sem a padronização adicional do projeto. KNNClassifier(standardize=True) calcula média e escala apenas no treino de cada fold. No ajuste final, recalcula ambos sobre as 7.352 amostras do desenvolvimento.')
p('dataprep.py verifica cobertura dos 30 sujeitos, presença das seis classes por sujeito, ausência de sobreposição de sujeitos e proporções de classe em cada partição. A tolerância de proporção é 0,03 (3 pontos percentuais em relação ao conjunto completo). Essa checagem consulta os rótulos globais, incluindo teste, mas não reamostra dados nem procura outra semente. As divisões não são estratificadas por classe; o agrupamento é por pessoa.')
h('Métricas e leitura dos resultados')
p('<b>Acurácia:</b> total de previsões corretas / total de amostras.<br/><b>Recall de uma classe:</b> acertos dessa classe / quantidade real dessa classe.<br/><b>Acurácia balanceada:</b> média aritmética dos recalls das seis classes; é a métrica global principal, com peso igual por atividade.<br/><b>Matriz de confusão:</b> linhas são classes reais e colunas são previsões, sempre na ordem 1 a 6; a diagonal contém os acertos.')
p('A escolha do Softmax minimiza o custo de validação (entropia cruzada + L2); a escolha do KNN maximiza a acurácia balanceada média dos folds. Logo, os critérios de seleção diferem, embora a métrica global de avaliação final seja comum. O KNN não otimiza uma função de custo por épocas, e não tem learning_rate, L2 ou tolerance.')
table([['Item','Escolha final','Acertos / 2.947','Acurácia','Balanc.'],*[ [k,('lr='+str(v['hiperparametros']['learning_rate'])+', L2=0' if 'learning_rate' in v['hiperparametros'] else 'k='+str(v['hiperparametros']['n_neighbors'])),str(sum(v['matriz_confusao_teste'][i][i] for i in range(6))),pct(v['acuracia_teste']),pct(v['acuracia_balanceada_teste'])] for k,v in M.items()]],[40,160,115,100,100])
h('Reprodução e arquivos de referência')
p('Fontes: enunciado Atividade_2_IA048_2s2026.pdf; src/dataprep.py, src/softmax.py e src/KNN.py; notebooks/model_training_1a.ipynb, model_traing_1b.ipynb, model_traing_2a.ipynb e model_traing_2b.ipynb. Os nomes “traing” reproduzem os arquivos existentes.')
p('Para cada modelo, results contém os arquivos *_busca.json (todas as tentativas), *_metricas.json (configuração, métricas e matriz de confusão) e *_final.npz (estado necessário às previsões). Prefixos: softmax_1a, knn_1b, softmax_2a e knn_2b. Softmax salva pesos, bias, classes, padronização, histórico de custo e sujeitos; KNN salva referências padronizadas, rótulos codificados, normas, classes, padronização e sujeitos. As previsões dos modelos recarregados foram verificadas. As execuções usaram OPENBLAS_NUM_THREADS=1; não foram medidos tempos para comparação de desempenho computacional.')

def results_tables(k):
 m=M[k];mat=m['matriz_confusao_teste'];names=['Caminhar','Subir escadas','Descer escadas','Sentado','Em pé','Deitado']
 h('Resultados por classe no teste oficial')
 table([['Classe real','Acertos','Total','Erros','Recall'],*[[f'{i+1} - {names[i]}',row[i],sum(row),sum(row)-row[i],pct(row[i]/sum(row))] for i,row in enumerate(mat)]],[195,75,75,75,95])
 p(f"<b>Acurácia:</b> {pct(m['acuracia_teste'])} • <b>Acurácia balanceada:</b> {pct(m['acuracia_balanceada_teste'])} • <b>Acertos:</b> {sum(mat[i][i] for i in range(6))}/2.947.")
 table([['Real / prevista','1','2','3','4','5','6'],*[[str(i+1),*row] for i,row in enumerate(mat)]],[95,70,70,70,70,70,70])

def softmax_section(k):
 page();m=M[k];b=B[k];hp=m['hiperparametros'];raw=k=='2a'
 title(f'{k} | Regressão logística Softmax', '768 valores dos sinais ACC/GYR' if raw else '561 atributos extraídos do UCI HAR')
 h('Modelo e otimização')
 p('Regressão logística multiclasse única com seis saídas Softmax; não se usa um-contra-um ou um-contra-todos. Logits = XW + b; previsão pela maior probabilidade. Classes originais 1 a 6 são codificadas internamente em 0 a 5. Em empate de probabilidade, argmax escolhe a primeira classe, isto é, o menor rótulo.')
 p('W e b começam em zero em cada fit; não há inicialização aleatória, warm start ou reaproveitamento de pesos entre tentativas. W tem dimensão '+('768 x 6' if raw else '561 x 6')+' e b tem seis componentes. Usa-se gradiente descendente full-batch, taxa constante, sem momentum, minibatches ou ponderação por classe. O Softmax subtrai o maior logit por amostra antes da exponenciação; probabilidades usadas no log são limitadas ao intervalo [1e-15, 1].')
 p('Custo J = média de -log(p da classe correta) + (L2 / 2) × soma dos quadrados de W. A regularização atua somente nos pesos; o bias não é penalizado. A parada ocorre quando |J atual - J anterior| é menor ou igual à tolerância, ou quando o limite de épocas é atingido. A tolerância usa o custo do treino antes da atualização; não é early stopping baseado na validação. O custo final é recalculado depois do ajuste. verbose=False.')
 h('Busca sequencial em duas etapas')
 table([['Etapa','Valores avaliados','Mantidos fixos'],['1: 20 combinações','learning_rate: 0.01, 0.03, 0.1, 0.2; L2: 0, 1e-5, 1e-4, 1e-3, 1e-2','max_epochs=1000; tolerance=1e-7'],['2: 9 combinações','max_epochs: 300, 800, 1500; tolerance: 1e-5, 1e-6, 1e-7','learning_rate e L2 vencedores da etapa 1']],[105,250,160])
 p('Cada tentativa treina nas 5.800 amostras e calcula model.loss nas 1.552 amostras de validação. A seleção usa min por custo; em empate exato, fica a primeira combinação da ordem das listas. O custo inclui L2 também na validação, portanto a penalização varia entre candidatos. Esta escolha não equivale a comparar apenas a entropia cruzada preditiva.')
 p('São 29 ajustes de seleção, seguidos de um ajuste final. Não se explora a grade conjunta de 180 combinações, nem se repete a primeira etapa após escolher épocas/tolerância. “Melhor” significa vencedor dessa busca sequencial. tune_softmax, que selecionaria por acurácia balanceada, não é usado pelos notebooks; eles chamam diretamente SoftmaxRegression.fit e loss.')
 h('Configuração escolhida e ajuste final')
 p(f"<b>learning_rate={hp['learning_rate']}; L2={hp['l2']}; max_epochs={hp['max_epochs']}; tolerance={hp['tolerance']}.</b> Um novo modelo é ajustado do zero com treino + validação (7.352 amostras), mantendo a padronização do treino interno. Executou {m['epocas_executadas']} épocas e parou pelo limite, sem atingir a tolerância. O teste oficial de 2.947 amostras é avaliado após a escolha.")
 p(f"Custo final no treino: <b>{num(m['custo_treino_final'])}</b>; custo no teste: <b>{num(m['custo_teste'])}</b>. L2=0 torna esses custos iguais à entropia cruzada média. A ausência de regularização é o resultado da busca, não uma premissa imposta.")
 page();title(f'{k} | Registro numérico', 'Custos de validação por combinação e avaliação final')
 h('Etapa 1: custo na validação (1.000 épocas; tolerância 1e-7)')
 lrs=[.01,.03,.1,.2];l2s=[0,1e-5,1e-4,1e-3,1e-2]
 table([['lr / L2',*map(str,l2s)],*[[str(lr),*[num(next(r['custo_validacao'] for r in b['etapa_1'] if r['learning_rate']==lr and r['l2']==l2)) for l2 in l2s]] for lr in lrs]],[65,90,90,90,90,90])
 h('Etapa 2: custo na validação com learning_rate e L2 escolhidos')
 ts=[1e-5,1e-6,1e-7]
 table([['Épocas / tolerância',*map(str,ts)],*[[str(e),*[num(next(r['custo_validacao'] for r in b['etapa_2'] if r['max_epochs']==e and r['tolerance']==t)) for t in ts]] for e in [300,800,1500]]],[155,120,120,120])
 p('Os valores são arredondados a seis casas apenas na exibição; a seleção usa os valores completos salvos no JSON. Tolerâncias distintas podem produzir custos idênticos quando não causam parada antecipada.','SmallX')
 results_tables(k)

def knn_section(k):
 page();m=M[k];b=B[k];raw=k=='2b'
 title(f'{k} | K-nearest neighbors', '768 valores dos sinais ACC/GYR' if raw else '561 atributos extraídos do UCI HAR')
 h('Modelo e parâmetros fixos')
 p('KNN multiclasse com distância euclidiana e voto majoritário uniforme: todos os vizinhos têm o mesmo peso. Em empate de votos, escolhe o menor rótulo. Não há treinamento iterativo, função de perda otimizada, épocas, taxa de aprendizado ou L2. fit calcula a padronização e armazena as amostras de referência e seus rótulos.')
 p('standardize=True no classificador; batch_size=256 limita o número de consultas processadas de cada vez e não é um hiperparâmetro estatístico otimizado. Distâncias são calculadas por normas ao quadrado e produto matricial; pequenos valores negativos numéricos são truncados em zero. Não se extrai raiz quadrada, pois a ordenação dos vizinhos é a mesma. A busca usa argpartition e ordenação estável dos candidatos; não há regra adicional por sujeito ou classe para distâncias exatamente iguais.')
 h('Escolha de k por validação cruzada')
 p('<b>Grade:</b> [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]. São 16 valores candidatos, não k=16. Os pares não foram testados. Esta grade é a mesma em 1b e 2b; a escolha fica restrita a ela.')
 p('Usa-se cross_validate_knn sobre todas as 7.352 amostras do treino oficial. Os 21 IDs ordenados são embaralhados com default_rng(42) e divididos por np.array_split em cinco grupos (5, 4, 4, 4 e 4 sujeitos). Cada grupo valida uma vez, e o complemento treina; a divisão não é estratificada. Os folds são idênticos em 1b e 2b.')
 p('Média e escala são estimadas apenas no treino de cada fold. Uma busca de até 31 vizinhos por fold é reutilizada para avaliar todos os k; assim há cinco ajustes e 80 avaliações (16 x 5). O critério é a maior média aritmética das cinco acurácias balanceadas, sem ponderar pelo tamanho dos folds. O desvio-padrão é populacional (ddof=0), não erro-padrão nem intervalo de confiança. Em empate de média, escolhe-se o menor k.')
 table([['Fold','Sujeitos de validação','Amostras treino','Amostras validação'],*[[f['fold'],', '.join(map(str,f['sujeitos_validacao'])),f['amostras_treino'],f['amostras_validacao']] for f in b['folds']]],[45,240,115,115])
 p('Os sujeitos de treino em cada fold são o complemento dos sujeitos de validação dentro da lista de 21 participantes do desenvolvimento registrada na seção Geral. A divisão fixa de quatro sujeitos produzida por prepare_data é reunida ao treino antes da validação cruzada; não funciona como holdout adicional para o KNN.','SmallX')
 h('Configuração escolhida e ajuste final')
 p(f"<b>k={m['hiperparametros']['n_neighbors']}</b>, com média de validação <b>{pct(m['media_validacao_cruzada'])}</b> e desvio-padrão de <b>{m['desvio_padrao_validacao_cruzada']*100:.2f} pontos percentuais</b>. Um novo KNN é ajustado nas 7.352 amostras, recalculando média e escala em todo o desenvolvimento. As 2.947 amostras do teste são transformadas com esses parâmetros e classificadas uma vez após a seleção, sem ajustar k pelo teste.")
 page();title(f'{k} | Registro numérico','Média e desvio dos cinco folds; resultados do teste oficial')
 table([['k','Média balanceada','Desvio (p.p.)','k','Média balanceada','Desvio (p.p.)'],*[[a['k'],pct(a['media_acuracia_balanceada']),f"{100*a['desvio_padrao']:.4f}",z['k'],pct(z['media_acuracia_balanceada']),f"{100*z['desvio_padrao']:.4f}"] for a,z in zip(b['resultados'][:8],b['resultados'][8:])]],[35,140,82.5,35,140,82.5])
 p('Resultados individuais de cada fold, sem arredondamento, estão em results/knn_'+k+'_busca.json. A média e o desvio acima descrevem a validação, não o teste.','SmallX')
 results_tables(k)
 p('Estado final salvo em results/knn_'+k+'_final.npz; métricas e matriz em results/knn_'+k+'_metricas.json. O histórico da busca permite verificar a escolha sem refazer o cálculo.','SmallX')

softmax_section('1a');knn_section('1b');softmax_section('2a');knn_section('2b')
SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=42,bottomMargin=51,title='Atividade 2 - Decisões dos modelos',author='Registro dos experimentos IA048').build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
