import tkinter as tk
from tkinter import messagebox
import heapq
from collections import deque
import random
import time

# -------------------- ESTADOS --------------------
ESTADO_OBJETIVO = [
    [1, 2, 3],
    [8, 0, 4],
    [7, 6, 5]
]

# -------------------- NÍVEIS DE DIFICULDADE --------------------
NIVEIS = {
    "Fácil": [
        [0, 2, 3],
        [1, 8, 4],
        [7, 6, 5]
    ],
    "Médio": [
        [2, 8, 3],
        [1, 6, 4],
        [7, 0, 5]
    ],
    "Difícil": [
        [5, 6, 7],
        [4, 0, 8],
        [3, 2, 1]
    ]
}

ESTADO_INICIAL = [linha[:] for linha in NIVEIS["Médio"]]

# -------------------- FUNÇÕES AUXILIARES --------------------
def encontrar_posicao(estado, valor):
    for i in range(3):
        for j in range(3):
            if estado[i][j] == valor:
                return i, j
    return None

def gerar_vizinhos(estado):
    i, j = encontrar_posicao(estado, 0)
    moves = []
    direcoes = [(-1,0), (1,0), (0,-1), (0,1)]
    
    for di, dj in direcoes:
        ni, nj = i + di, j + dj
        if 0 <= ni < 3 and 0 <= nj < 3:
            novo_estado = [linha[:] for linha in estado]
            novo_estado[i][j], novo_estado[ni][nj] = novo_estado[ni][nj], novo_estado[i][j]
            moves.append(novo_estado)
    return moves

def estado_para_tupla(estado):
    return tuple(tuple(linha) for linha in estado)

# -------------------- HEURÍSTICA (A*) --------------------
def manhattan(estado):
    dist = 0
    for i in range(3):
        for j in range(3):
            valor = estado[i][j]
            if valor != 0:
                i_obj, j_obj = encontrar_posicao(ESTADO_OBJETIVO, valor)
                dist += abs(i - i_obj) + abs(j - j_obj)
    return dist

# -------------------- BUSCA EM LARGURA --------------------
def busca_em_largura():
    fila = deque([(ESTADO_INICIAL, [])])
    visitados = set()

    while fila:
        estado, caminho = fila.popleft()
        if estado == ESTADO_OBJETIVO:
            return caminho + [estado]
        
        visitados.add(estado_para_tupla(estado))
        
        for vizinho in gerar_vizinhos(estado):
            if estado_para_tupla(vizinho) not in visitados:
                fila.append((vizinho, caminho + [estado]))
    return None

# -------------------- BUSCA A* --------------------
def busca_a_estrela():
    fila = []
    heapq.heappush(fila, (manhattan(ESTADO_INICIAL), 0, ESTADO_INICIAL, []))
    visitados = set()

    while fila:
        _, g, estado, caminho = heapq.heappop(fila)
        if estado == ESTADO_OBJETIVO:
            return caminho + [estado]
        
        visitados.add(estado_para_tupla(estado))
        
        for vizinho in gerar_vizinhos(estado):
            if estado_para_tupla(vizinho) not in visitados:
                novo_caminho = caminho + [estado]
                f = g + 1 + manhattan(vizinho)
                heapq.heappush(fila, (f, g + 1, vizinho, novo_caminho))
    return None

def comparar_buscas():
    import time

    print("Comparando Busca em Largura vs A*...")
    for nome, func in [("Largura", busca_em_largura), ("A*", busca_a_estrela)]:
        inicio = time.time()
        caminho = func()
        duracao = time.time() - inicio
        print(f"{nome}: {len(caminho)-1} passos | {duracao:.4f}s")

# -------------------- INTERFACE GRÁFICA --------------------
class JogoOito:
    def __init__(self, root):
        self.root = root
        self.root.title("🧩 Jogo dos Oito - Busca em Largura e A*")
        self.root.configure(bg="#EBF5FB")

        # Layout principal
        self.container = tk.Frame(root, bg="#EBF5FB")
        self.container.pack(padx=20, pady=20)

        # Tabuleiro à esquerda
        self.tabuleiro = tk.Frame(self.container, bg="#EBF5FB")
        self.tabuleiro.grid(row=0, column=0, padx=30)

        self.botoes = [[None for _ in range(3)] for _ in range(3)]
        self.estado_atual = [linha[:] for linha in ESTADO_INICIAL]

        for i in range(3):
            for j in range(3):
                valor = self.estado_atual[i][j]
                btn = tk.Button(self.tabuleiro,
                                text=str(valor) if valor != 0 else "",
                                width=6, height=3,
                                font=("Arial", 22, "bold"),
                                bg="#AED6F1", fg="#1B2631",
                                relief="raised",
                                command=lambda x=i, y=j: self.mover_bloco(x, y))
                btn.grid(row=i, column=j, padx=5, pady=5)
                self.botoes[i][j] = btn

        # Painel lateral à direita
        self.painel = tk.Frame(self.container, bg="#D6EAF8", bd=2, relief="groove")
        self.painel.grid(row=0, column=1, sticky="ns")

        tk.Label(self.painel, text="🎮 Controle", bg="#D6EAF8",
                 font=("Arial", 14, "bold")).pack(pady=10)

        # Botões de dificuldade
        tk.Label(self.painel, text="Selecione a dificuldade:",
                 bg="#D6EAF8", font=("Arial", 12, "bold")).pack(pady=5)

        self.dificuldade = "Médio"
        for nivel, cor in [("Fácil", "#58D68D"), ("Médio", "#F4D03F"), ("Difícil", "#EC7063")]:
            tk.Button(self.painel, text=nivel, bg=cor, fg="black", width=10,
                      font=("Arial", 12, "bold"),
                      command=lambda n=nivel: self.definir_dificuldade(n)).pack(pady=5)

        # Botões principais
        tk.Button(self.painel, text="🔄 Embaralhar", command=self.embaralhar,
                  bg="#5DADE2", font=("Arial", 12, "bold"), width=15).pack(pady=5)
        tk.Button(self.painel, text="🌐 Resolver (Amplitude)", command=self.executar_largura,
                  bg="#85C1E9", font=("Arial", 12, "bold"), width=20).pack(pady=5)
        tk.Button(self.painel, text="⭐ Resolver (A*)", command=self.executar_a_estrela,
                  bg="#58D68D", font=("Arial", 12, "bold"), width=20).pack(pady=5)
        tk.Button(self.painel, text="⏹ Resetar", command=self.resetar,
                  bg="#F1948A", font=("Arial", 12, "bold"), width=15).pack(pady=5)

        # Informações
        self.status = tk.Label(self.painel, text="Dificuldade atual: Médio",
                               bg="#D6EAF8", font=("Arial", 11, "italic"))
        self.status.pack(pady=10)

    # ------------ Funções de interface ------------
    def atualizar_tabuleiro(self, estado):
        for i in range(3):
            for j in range(3):
                valor = estado[i][j]
                self.botoes[i][j]["text"] = str(valor) if valor != 0 else ""
                self.botoes[i][j]["bg"] = "#AED6F1" if valor != 0 else "#EBF5FB"
        self.root.update()

    def mover_bloco(self, i, j):
        i0, j0 = encontrar_posicao(self.estado_atual, 0)
        if abs(i - i0) + abs(j - j0) == 1:
            self.estado_atual[i0][j0], self.estado_atual[i][j] = self.estado_atual[i][j], self.estado_atual[i0][j0]
            self.atualizar_tabuleiro(self.estado_atual)
            if self.estado_atual == ESTADO_OBJETIVO:
                messagebox.showinfo("🎉 Parabéns!", "Você resolveu o quebra-cabeça!")

    def definir_dificuldade(self, nivel):
        self.dificuldade = nivel
        self.estado_atual = [linha[:] for linha in NIVEIS[nivel]]
        global ESTADO_INICIAL
        ESTADO_INICIAL = [linha[:] for linha in NIVEIS[nivel]]
        self.atualizar_tabuleiro(self.estado_atual)
        self.status.config(text=f"Dificuldade atual: {nivel}")

    def embaralhar(self):
        for _ in range(50):
            self.estado_atual = random.choice(gerar_vizinhos(self.estado_atual))
        global ESTADO_INICIAL
        ESTADO_INICIAL = [linha[:] for linha in self.estado_atual]
        self.atualizar_tabuleiro(self.estado_atual)
        self.status.config(text=f"🔄 Tabuleiro embaralhado ({self.dificuldade})")

    def resetar(self):
        self.definir_dificuldade(self.dificuldade)

    def animar_caminho(self, caminho):
        if not caminho:
            messagebox.showinfo("😕", "Nenhum caminho encontrado!")
            return
        
        inicio = time.time()
        def animar(i=0):
            if i < len(caminho):
                self.atualizar_tabuleiro(caminho[i])
                self.root.after(400, animar, i + 1)
            else:
                fim = time.time() - inicio
                self.status.config(text=f"✅ Resolvido ({len(caminho)-1} passos, {fim:.2f}s)")
                messagebox.showinfo("Concluído", "Objetivo alcançado!")
        animar()

    def executar_largura(self):
        self.status.config(text="🔍 Executando busca em largura...")
        caminho = busca_em_largura()
        self.animar_caminho(caminho)

    def executar_a_estrela(self):
        self.status.config(text="🔍 Executando busca A*...")
        caminho = busca_a_estrela()
        self.animar_caminho(caminho)

# -------------------- EXECUÇÃO --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = JogoOito(root)
    root.mainloop()
    comparar_buscas()