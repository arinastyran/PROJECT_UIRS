import pickle
import numpy as np
from scipy.fft import dct

# Класс DCT (нужен для pickle)
class DCT:
    def __init__(self, data, cutoff_amount=None, range_min=None, range_max=None):
        self.range_min = range_min
        self.range_max = range_max
        self.N = len(data)
        self.coefficients = dct(data, type=2)
        
        if cutoff_amount is not None:
            indices = sorted(range(len(self.coefficients)), 
                           key=lambda i: abs(self.coefficients[i]), 
                           reverse=True)
            for idx in indices[cutoff_amount:]:
                self.coefficients[idx] = 0

# Загрузка моделей
with open(r"C:\UIRS\surface-classification\dct_models.pkl", 'rb') as file:
    dct_models, _ = pickle.load(file)

surfaces = sorted(list(dct_models.keys()))

print("АНАЛИТИЧЕСКИЕ ФОРМУЛЫ DCT-2 МОДЕЛЕЙ (LaTeX формат)")

for surf in surfaces:
    model = dct_models[surf]
    N = model.N
    coeffs = model.coefficients
    
    # Собираем ненулевые члены
    terms = []
    
    # Нулевой коэффициент (постоянная составляющая)
    if coeffs[0] != 0:
        terms.append(f"\\frac{{{coeffs[0]:.4f}}}{{2}}")
    
    # Остальные коэффициенты
    for n in range(1, N):
        if coeffs[n] != 0:
            # Сохраняем знак ВНУТРИ числа (LaTeX сам корректно отрисует минус)
            terms.append(f"{coeffs[n]:.4f} \\cos\\left(\\frac{{{n}\\pi}}{{{N}}}(\\alpha + 0.5)\\right)")
    
    # Соединяем через +
    terms_str = " + ".join(terms)
    
    # Имя поверхности
    surf_latex = surf.replace('_', '\\_')
    
    latex_formula = f"f^{{\\text{{{surf_latex}}}}}(\\alpha) = \\frac{{1}}{{{N}}} \\left( {terms_str} \\right)"
    
    print(f"\nПоверхность: {surf.replace('_', ' ').title()}")
    print(latex_formula)
