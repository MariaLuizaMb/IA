import random
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "Gráficos AG"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# Funções externas
# ------------------------------------------------------------

def binario(num, num_bits):
    """Converte inteiro para vetor binário com sinal."""
    return list(bin(num).replace("0b", "" if num < 0 else "+").zfill(num_bits))

def binario_para_int(vetor):
    """Transforma vetor binário em inteiro."""
    return int("".join(vetor), 2)

def funcao_objetivo(x):
    """Função f(x) = x² - 3x + 4."""
    return x**2 - 3*x + 4


# ------------------------------------------------------------
# Classe do Algoritmo Genético
# ------------------------------------------------------------

class AlgoritmoGenetico:
    def __init__(self, x_min, x_max, tam_pop, taxa_mutacao, taxa_crossover, num_geracoes):
        self.x_min = x_min
        self.x_max = x_max
        self.tam_pop = tam_pop
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.num_geracoes = num_geracoes

        bits_min = len(bin(x_min).replace("0b", "" if x_min < 0 else "+"))
        bits_max = len(bin(x_max).replace("0b", "" if x_max < 0 else "+"))
        self.num_bits = max(bits_min, bits_max)

        self.populacao = self.gerar_populacao()
        self.avaliacoes = []

    def gerar_populacao(self):
        """Gera população inicial."""
        populacao = []
        for _ in range(self.tam_pop):
            n = random.randint(self.x_min, self.x_max)
            populacao.append(binario(n, self.num_bits))
        return populacao

    def avaliar(self):
        """Avalia indivíduos."""
        self.avaliacoes = []
        for ind in self.populacao:
            x = binario_para_int(ind)
            self.avaliacoes.append(funcao_objetivo(x))

    def selecionar_por_torneio(self):
        """Seleção por torneio (2 indivíduos)."""
        i1 = random.randint(0, self.tam_pop - 1)
        i2 = random.randint(0, self.tam_pop - 1)
        return self.populacao[i1] if self.avaliacoes[i1] >= self.avaliacoes[i2] else self.populacao[i2]

    def ajustar_limites(self, individuo):
        """Mantém indivíduo dentro dos limites."""
        valor = binario_para_int(individuo)
        if valor < self.x_min:
            return binario(self.x_min, self.num_bits)
        if valor > self.x_max:
            return binario(self.x_max, self.num_bits)
        return individuo

    def crossover(self, pai, mae):
        """Crossover de 1 ponto."""
        if random.randint(1, 100) <= self.taxa_crossover:
            ponto = random.randint(1, self.num_bits - 1)
            f1 = pai[:ponto] + mae[ponto:]
            f2 = mae[:ponto] + pai[ponto:]
        else:
            f1, f2 = pai[:], mae[:]

        return self.ajustar_limites(f1), self.ajustar_limites(f2)

    def mutar(self, individuo):
        """Mutação de 1 bit."""
        if random.randint(1, 100) <= self.taxa_mutacao:
            pos = random.randint(0, self.num_bits - 1)
            individuo[pos] = '1' if individuo[pos] == '0' else '0'

        return self.ajustar_limites(individuo)

    def melhor_individuo(self):
        melhor_idx = self.avaliacoes.index(max(self.avaliacoes))
        return self.populacao[melhor_idx], self.avaliacoes[melhor_idx]


# ------------------------------------------------------------
# Execução do AG e geração de gráfico
# ------------------------------------------------------------

def executar_ag(tam_pop, geracoes, taxa_mut, taxa_cross, numero_execucao):
    ag = AlgoritmoGenetico(-10, 10, tam_pop, taxa_mut, taxa_cross, geracoes)

    historico = []

    ag.avaliar()

    for g in range(geracoes):
        melhor, fit = ag.melhor_individuo()
        historico.append(fit)

        nova_pop = []

        while len(nova_pop) < tam_pop:
            pai = ag.selecionar_por_torneio()
            mae = ag.selecionar_por_torneio()

            f1, f2 = ag.crossover(pai, mae)
            f1 = ag.mutar(f1)
            f2 = ag.mutar(f2)

            nova_pop.append(f1)
            if len(nova_pop) < tam_pop:
                nova_pop.append(f2)

        ag.populacao = nova_pop
        ag.avaliar()

    melhor, fit = ag.melhor_individuo()
    historico.append(fit)

    print(f"\n→ Execução {numero_execucao}")
    print("Melhor indivíduo final:", melhor, "Fitness =", fit)

    plt.figure(figsize=(9, 6))
    plt.plot(historico, marker='o')
    plt.title(
        f"Evolução do Melhor Fitness\n"
        f"(Pop: {tam_pop} | Gerações: {geracoes} | Mutação: {taxa_mut}% | Crossover: {taxa_cross}%)"
    )
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness")
    plt.grid(True)

    nome_arquivo = f"evolucao_ag_execucao_{numero_execucao}.png"
    plt.savefig(os.path.join(RESULTS_DIR,nome_arquivo))
    plt.show()

    print(f"Gráfico salvo como: {nome_arquivo}\n")


# ------------------------------------------------------------
# Loop de Execuções com Entrada do Usuário
# ------------------------------------------------------------

def main():
    # Valores iniciais (primeira execução obrigatória)
    tam_pop = 4
    geracoes = 5
    taxa_mut = 1
    taxa_cross = 70

    numero_execucao = 1

    # Executa primeiro com valores padrão
    executar_ag(tam_pop, geracoes, taxa_mut, taxa_cross, numero_execucao)

    # Loop para o usuário alterar parâmetros
    while True:
        print("\nDigite 'continuar' para inserir novos parâmetros ou 'sair' para encerrar.")

        entrada = input("Deseja continuar? ").strip().lower()
        if entrada == "sair":
            print("Encerrando o programa...")
            break
        elif entrada == "continuar":
            # Coleta novos parâmetros
            print("Insira os novos parâmetros:")
            print("Tamanho atual da população", tam_pop)
            tam_pop = int(input("Tamanho da população (4-30): "))
            print("Número atual de gerações", geracoes)
            geracoes = int(input("Número de gerações (5-20): "))
            print("Taxa atual de mutação (%)", taxa_mut)
            taxa_mut = int(input("Taxa de mutação (%): "))
            print("Taxa atual de crossover (%)", taxa_cross)
            taxa_cross = int(input("Taxa de crossover (%): "))

            numero_execucao += 1
            executar_ag(tam_pop, geracoes, taxa_mut, taxa_cross, numero_execucao)


if __name__ == "__main__":
    main()