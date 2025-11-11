import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sympy as sp
from sympy import symbols, diff, integrate, solve, limit, oo, lambdify
from scipy.optimize import minimize
from scipy.integrate import dblquad, tplquad
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Calculadora Cálculo Multivariable",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 2rem;
}
.info-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f0f2f6;
    border-left: 5px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🧮 Calculadora de Cálculo Multivariable</h1>', unsafe_allow_html=True)

# Sidebar para navegación
st.sidebar.title("📚 Módulos")
modulo = st.sidebar.selectbox(
    "Selecciona el módulo:",
    [
        "🏠 Inicio",
        "📊 Visualización de Funciones",
        "🔍 Dominio y Rango",
        "📐 Derivadas Parciales",
        "🎯 Optimización con Restricciones",
        "∫ Integrales Múltiples",
        "📖 Ayuda y Ejemplos"
    ]
)

def parse_function(func_str, variables):
    """Convierte string a función simbólica"""
    try:
        # Reemplazar algunos símbolos comunes
        func_str = func_str.replace('^', '**')
        func_str = func_str.replace('sen', 'sin')
        func_str = func_str.replace('ln', 'log')
        
        if len(variables) == 2:
            x, y = variables
            return sp.sympify(func_str), lambdify([x, y], sp.sympify(func_str), 'numpy')
        elif len(variables) == 3:
            x, y, z = variables
            return sp.sympify(func_str), lambdify([x, y, z], sp.sympify(func_str), 'numpy')
    except Exception as e:
        st.error(f"Error al interpretar la función: {e}")
        return None, None

def plot_3d_surface(func_numeric, x_range, y_range, title="Superficie 3D"):
    """Crear gráfico 3D con Plotly"""
    x_vals = np.linspace(x_range[0], x_range[1], 50)
    y_vals = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    try:
        Z = func_numeric(X, Y)
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            opacity=0.9
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=800,
            height=600
        )
        return fig
    except Exception as e:
        st.error(f"Error al graficar: {e}")
        return None

def plot_contour(func_numeric, x_range, y_range, title="Curvas de Nivel"):
    """Crear gráfico de curvas de nivel"""
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    y_vals = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    try:
        Z = func_numeric(X, Y)
        
        fig = go.Figure(data=go.Contour(
            x=x_vals,
            y=y_vals,
            z=Z,
            colorscale='Viridis',
            contours=dict(showlabels=True)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='X',
            yaxis_title='Y',
            width=600,
            height=500
        )
        return fig
    except Exception as e:
        st.error(f"Error al graficar curvas de nivel: {e}")
        return None

# MÓDULO INICIO
if modulo == "🏠 Inicio":
    st.markdown("""
    ## ¡Bienvenido a la Calculadora de Cálculo Multivariable! 👋
    
    Esta aplicación te permite explorar y calcular conceptos fundamentales del cálculo multivariable:
    
    ### 🎯 Funcionalidades Principales:
    - **Visualización 3D**: Gráficos interactivos de superficies y curvas de nivel
    - **Derivadas Parciales**: Cálculo automático y visualización de gradientes
    - **Optimización**: Multiplicadores de Lagrange y puntos críticos
    - **Integrales Múltiples**: Cálculo de volúmenes, masas y centroides
    - **Análisis de Dominio**: Determinación automática de dominios y rangos
    
    ### 🚀 Cómo empezar:
    1. Selecciona un módulo en la barra lateral
    2. Ingresa tu función matemática
    3. Explora los resultados interactivos
    """)
    
    # Ejemplos rápidos
    st.markdown('<h2 class="sub-header">📝 Ejemplos Rápidos</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Funciones de ejemplo:**
        - `x**2 + y**2` (Paraboloide)
        - `sin(x) * cos(y)` (Función trigonométrica)
        - `x**2 - y**2` (Silla de montar)
        - `exp(-(x**2 + y**2))` (Gaussiana)
        """)
    
    with col2:
        st.markdown("""
        **Sintaxis válida:**
        - Use `**` para potencias: `x**2`
        - Funciones: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`
        - Constantes: `pi`, `e`
        """)

# MÓDULO VISUALIZACIÓN
elif modulo == "📊 Visualización de Funciones":
    st.markdown('<h2 class="sub-header">📊 Visualización de Funciones de Dos Variables</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Configuración")
        func_input = st.text_input("Función f(x,y):", value="x**2 + y**2", help="Ejemplo: x**2 + y**2")
        
        st.markdown("**Rango de visualización:**")
        x_min = st.number_input("X mínimo:", value=-5.0)
        x_max = st.number_input("X máximo:", value=5.0)
        y_min = st.number_input("Y mínimo:", value=-5.0)
        y_max = st.number_input("Y máximo:", value=5.0)
        
        plot_type = st.selectbox("Tipo de gráfico:", ["Superficie 3D", "Curvas de Nivel", "Ambos"])
    
    with col2:
        if func_input:
            x, y = symbols('x y')
            func_sym, func_num = parse_function(func_input, [x, y])
            
            if func_sym and func_num:
                st.markdown(f"**Función:** f(x,y) = {func_sym}")
                
                if plot_type in ["Superficie 3D", "Ambos"]:
                    fig_3d = plot_3d_surface(func_num, (x_min, x_max), (y_min, y_max))
                    if fig_3d:
                        st.plotly_chart(fig_3d, use_container_width=True)
                
                if plot_type in ["Curvas de Nivel", "Ambos"]:
                    fig_contour = plot_contour(func_num, (x_min, x_max), (y_min, y_max))
                    if fig_contour:
                        st.plotly_chart(fig_contour, use_container_width=True)

# MÓDULO DOMINIO Y RANGO
elif modulo == "🔍 Dominio y Rango":
    st.markdown('<h2 class="sub-header">🔍 Análisis de Dominio y Rango</h2>', unsafe_allow_html=True)
    
    func_input = st.text_input("Función f(x,y):", value="log(x - y**2)", help="Ejemplo: log(x - y**2)")
    
    if func_input:
        x, y = symbols('x y')
        func_sym, func_num = parse_function(func_input, [x, y])
        
        if func_sym:
            st.markdown(f"**Función analizada:** f(x,y) = {func_sym}")
            
            def analyze_domain_symbolic(func):
                """Análisis simbólico del dominio"""
                domain_conditions = []
                domain_description = []
                
                # Encontrar todas las subexpresiones problemáticas
                subexprs = list(func.atoms())
                
                # Analizar logaritmos
                for expr in sp.preorder_traversal(func):
                    if expr.func == sp.log:
                        arg = expr.args[0]
                        domain_conditions.append(f"{arg} > 0")
                        domain_description.append(f"**Logaritmo:** El argumento {arg} debe ser estrictamente positivo")
                
                # Analizar raíces cuadradas
                for expr in sp.preorder_traversal(func):
                    if expr.func == sp.sqrt:
                        arg = expr.args[0]
                        domain_conditions.append(f"{arg} >= 0")
                        domain_description.append(f"**Raíz cuadrada:** El argumento {arg} debe ser no negativo")
                
                # Analizar divisiones
                for expr in sp.preorder_traversal(func):
                    if expr.is_Pow and expr.args[1] < 0:
                        base = expr.args[0]
                        domain_conditions.append(f"{base} ≠ 0")
                        domain_description.append(f"**División:** El denominador {base} no puede ser cero")
                
                # Analizar funciones trigonométricas inversas
                for expr in sp.preorder_traversal(func):
                    if expr.func in [sp.asin, sp.acos]:
                        arg = expr.args[0]
                        domain_conditions.append(f"-1 <= {arg} <= 1")
                        domain_description.append(f"**Función inversa:** {expr.func.__name__}({arg}) requiere -1 ≤ {arg} ≤ 1")
                
                return domain_conditions, domain_description
            
            def analyze_range_symbolic(func, domain_conditions):
                """Análisis simbólico del rango"""
                range_analysis = []
                
                try:
                    # Para logaritmos: rango (-∞, +∞)
                    if any('log' in str(expr) for expr in sp.preorder_traversal(func)):
                        range_analysis.append("**Logaritmo presente:** El rango puede extenderse a (-∞, +∞)")
                    
                    # Para funciones cuadráticas
                    try:
                        if func.is_polynomial(x, y):
                            # Encontrar el grado sin usar sp.degree()
                            func_expanded = sp.expand(func)
                            max_degree = 0
                            for term in sp.Add.make_args(func_expanded):
                                degree_x = sp.degree(term, x) if x in term.free_symbols else 0
                                degree_y = sp.degree(term, y) if y in term.free_symbols else 0
                                total_degree = degree_x + degree_y
                                max_degree = max(max_degree, total_degree)
                            
                            if max_degree == 2:
                                range_analysis.append("**Función cuadrática:** Analizar vértice para extremos")
                    except:
                        pass
                    
                    # Para exponenciales: siempre positivo
                    if any(expr.func == sp.exp for expr in sp.preorder_traversal(func)):
                        range_analysis.append("**Exponencial presente:** Rango siempre positivo (0, +∞)")
                    
                except Exception as e:
                    pass
                
                return range_analysis
            
            # Realizar análisis simbólico
            domain_conditions, domain_descriptions = analyze_domain_symbolic(func_sym)
            range_analysis = analyze_range_symbolic(func_sym, domain_conditions)
            
            # Mostrar análisis del dominio
            st.markdown("### 📍 **Dominio**")
            
            if domain_conditions:
                st.markdown("**Restricciones identificadas:**")
                for desc in domain_descriptions:
                    st.write(f"• {desc}")
                
                st.markdown("**Condiciones del dominio:**")
                for i, condition in enumerate(domain_conditions, 1):
                    st.latex(f"\\text{{{i}.}} \\quad {condition}")
                
                # Intentar resolver el sistema de inecuaciones
                st.markdown("**Interpretación geométrica:**")
                
                # Casos específicos comunes
                func_str = str(func_sym)
                
                if 'log(x - y**2)' in func_str:
                    st.markdown("**Solución de la inecuación:**")
                    st.latex(r"x - y^2 > 0 \quad \Rightarrow \quad x > y^2")
                    st.markdown("**Dominio:**")
                    st.latex(r"D = \{(x,y) \in \mathbb{R}^2 \mid x > y^2\}")
                    st.markdown("**Descripción:** Todo el plano a la derecha de la parábola x = y² (la parábola misma está excluida).")
                
                elif 'sqrt(' in func_str:
                    # Para raíces cuadradas
                    if 'x**2 + y**2 - ' in func_str:
                        # Círculo
                        try:
                            # Extraer el radio del término constante
                            import re
                            match = re.search(r'x\*\*2 \+ y\*\*2 - (\d+)', func_str)
                            if match:
                                r_sq = match.group(1)
                                r = sp.sqrt(int(r_sq))
                                st.latex(f"x^2 + y^2 \\geq {r_sq} \\quad \\Rightarrow \\quad x^2 + y^2 \\geq {r}^2")
                                st.markdown(f"**Dominio:** Exterior del círculo de radio {r} (incluye la circunferencia)")
                        except:
                            st.markdown("Región donde el argumento de la raíz es no negativo")
                    else:
                        st.markdown("Región donde el argumento de la raíz cuadrada es no negativo")
                
                elif '1/' in func_str or 'log(' in func_str:
                    st.markdown("Región donde se evitan las singularidades (divisiones por cero o argumentos no válidos)")
                
            else:
                st.success("✅ **La función está definida para todos los números reales**")
                st.latex(r"D = \mathbb{R}^2")
            
            # Mostrar análisis del rango
            st.markdown("### 📏 **Rango**")
            
            # Análisis específico por tipo de función
            func_str = str(func_sym)
            
            if 'log(x - y**2)' in func_str:
                st.markdown("**Análisis del comportamiento:**")
                st.markdown("El argumento x - y² puede tomar cualquier valor positivo:")
                st.markdown("• Para un y fijo, se puede escoger x arbitrariamente cercano a y² (pero mayor) para que x - y² → 0⁺")
                st.markdown("• Para un y fijo, se puede escoger x grande para que x - y² → +∞")
                
                st.markdown("**Comportamiento límite:**")
                st.latex(r"\text{Cuando } x - y^2 \to 0^+ \text{, entonces } \ln(x - y^2) \to -\infty")
                st.latex(r"\text{Cuando } x - y^2 \to +\infty \text{, entonces } \ln(x - y^2) \to +\infty")
                
                st.markdown("**Por tanto:**")
                st.latex(r"\text{Rango} = (-\infty, +\infty)")
                
            elif 'sqrt(' in func_str:
                st.markdown("**Para funciones con raíz cuadrada:**")
                if 'x**2 + y**2' in func_str:
                    st.markdown("• El mínimo valor se alcanza cuando el argumento de la raíz es mínimo")
                    st.markdown("• El rango típicamente es [valor_mínimo, +∞)")
                st.latex(r"\text{Rango} = [0, +\infty) \text{ (típico para raíces)}")
                
            elif func_sym.is_polynomial(x, y):
                try:
                    # Calcular el grado de manera segura
                    func_expanded = sp.expand(func_sym)
                    max_degree = 0
                    for term in sp.Add.make_args(func_expanded):
                        degree_x = sp.degree(term, x) if x in term.free_symbols else 0
                        degree_y = sp.degree(term, y) if y in term.free_symbols else 0
                        total_degree = degree_x + degree_y
                        max_degree = max(max_degree, total_degree)
                    
                    if max_degree == 2:
                        st.markdown("**Función cuadrática (grado 2):**")
                        st.markdown("• Puede tener un mínimo o máximo global")
                        st.markdown("• El rango puede ser [mínimo, +∞) o (-∞, máximo]")
                        
                        # Intentar encontrar puntos críticos
                        try:
                            fx_diff = diff(func_sym, x)
                            fy_diff = diff(func_sym, y)
                            critical_points = solve([fx_diff, fy_diff], [x, y])
                            
                            if critical_points:
                                st.markdown("**Puntos críticos encontrados:**")
                                if isinstance(critical_points, list):
                                    for pt in critical_points:
                                        if len(pt) == 2:
                                            x_val, y_val = pt
                                            f_val = func_sym.subs([(x, x_val), (y, y_val)])
                                            st.write(f"• Punto crítico: ({x_val}, {y_val}), f = {f_val}")
                                elif isinstance(critical_points, dict):
                                    x_val = critical_points.get(x, 0)
                                    y_val = critical_points.get(y, 0)
                                    f_val = func_sym.subs([(x, x_val), (y, y_val)])
                                    st.write(f"• Punto crítico: ({x_val}, {y_val}), f = {f_val}")
                        except:
                            pass
                    else:
                        st.markdown(f"**Función polinomial de grado {max_degree}:**")
                        st.latex(r"\text{Rango} = (-\infty, +\infty) \text{ (típico para polinomios)}")
                except:
                    st.markdown("**Función polinomial:**")
                    st.latex(r"\text{Rango} = (-\infty, +\infty)")
                            
            elif 'exp(' in func_str:
                st.markdown("**Función exponencial:**")
                st.latex(r"\text{Rango} = (0, +\infty)")
                st.markdown("Las exponenciales son siempre positivas")
                
            else:
                st.markdown("**Análisis general:**")
                if range_analysis:
                    for analysis in range_analysis:
                        st.write(f"• {analysis}")
                else:
                    st.info("💡 Realiza análisis límite o encuentra extremos para determinar el rango exacto")
            
            # Visualización
            st.markdown("### 🎨 **Visualización**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico 3D si es posible
                try:
                    fig_3d = plot_3d_surface(func_num, (-3, 5), (-3, 3), f"f(x,y) = {func_sym}")
                    if fig_3d:
                        st.plotly_chart(fig_3d, use_container_width=True)
                except Exception as e:
                    st.warning(f"No se pudo graficar la superficie 3D: {e}")
            
            with col2:
                # Mapa del dominio
                try:
                    x_range = np.linspace(-2, 5, 100)
                    y_range = np.linspace(-3, 3, 100)
                    X, Y = np.meshgrid(x_range, y_range)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        Z = func_num(X, Y)
                    
                    # Crear máscara del dominio
                    domain_mask = np.isfinite(Z) & ~np.isnan(Z)
                    
                    fig = go.Figure(data=go.Heatmap(
                        x=x_range,
                        y=y_range,
                        z=domain_mask.astype(int),
                        colorscale=[[0, 'red'], [1, 'blue']],
                        colorbar=dict(title="Dominio", tickvals=[0, 1], ticktext=["No definida", "Definida"])
                    ))
                    
                    # Agregar la curva frontera si es x > y²
                    if 'x - y**2' in func_str:
                        y_boundary = np.linspace(-3, 3, 100)
                        x_boundary = y_boundary**2
                        fig.add_trace(go.Scatter(
                            x=x_boundary, y=y_boundary,
                            mode='lines', name='Frontera: x = y²',
                            line=dict(color='yellow', width=3)
                        ))
                    
                    fig.update_layout(
                        title="Región del Dominio",
                        xaxis_title="x", yaxis_title="y",
                        width=500, height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error en visualización del dominio: {e}")
            
            # Ejemplos adicionales
            st.markdown("### 📚 **Ejemplos para probar:**")
            
            examples = {
                "log(x - y**2)": "Parábola como frontera",
                "sqrt(4 - x**2 - y**2)": "Interior de círculo",
                "1/(x**2 + y**2 - 1)": "Exterior de círculo",
                "log(x**2 + y**2)": "Todo el plano excepto origen",
                "sqrt(x + y)": "Región x + y ≥ 0",
                "x**2 + y**2": "Paraboloide (todo R²)"
            }
            
            selected_example = st.selectbox("Selecciona un ejemplo:", list(examples.keys()))
            if st.button("Usar ejemplo"):
                st.experimental_rerun()

# MÓDULO DERIVADAS PARCIALES
elif modulo == "📐 Derivadas Parciales":
    st.markdown('<h2 class="sub-header">📐 Derivadas Parciales y Gradientes</h2>', unsafe_allow_html=True)
    
    func_input = st.text_input("Función f(x,y):", value="x**2 - y**2", help="Ejemplo: x**2 - y**2")
    
    if func_input:
        x, y = symbols('x y')
        func_sym, func_num = parse_function(func_input, [x, y])
        
        if func_sym:
            # Calcular derivadas parciales
            fx = diff(func_sym, x)
            fy = diff(func_sym, y)
            
            # Sección de cálculos
            st.markdown("### 🧮 Cálculos Simbólicos")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Función:**")
                st.latex(f"f(x,y) = {sp.latex(func_sym)}")
            
            with col2:
                st.markdown(f"**Derivadas parciales:**")
                st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(fx)}")
                st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {sp.latex(fy)}")
            
            with col3:
                st.markdown(f"**Gradiente:**")
                st.latex(f"\\nabla f = ({sp.latex(fx)}, {sp.latex(fy)})")
            
            # Evaluación en punto específico
            st.markdown("### 📍 Evaluación en Punto Específico")
            
            col_a, col_b = st.columns([1, 2])
            
            with col_a:
                px = st.number_input("Coordenada x:", value=1.0, key="px")
                py = st.number_input("Coordenada y:", value=1.0, key="py")
                
                try:
                    f_val = float(func_sym.subs([(x, px), (y, py)]))
                    fx_val = float(fx.subs([(x, px), (y, py)]))
                    fy_val = float(fy.subs([(x, px), (y, py)]))
                    
                    st.markdown("**Valores en el punto:**")
                    st.write(f"• f({px}, {py}) = **{f_val:.4f}**")
                    st.write(f"• ∂f/∂x({px}, {py}) = **{fx_val:.4f}**")
                    st.write(f"• ∂f/∂y({px}, {py}) = **{fy_val:.4f}**")
                    
                    magnitud = np.sqrt(fx_val**2 + fy_val**2)
                    st.write(f"• ||∇f|| = **{magnitud:.4f}**")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
            
            with col_b:
                st.markdown("**Interpretación del gradiente:**")
                st.info("""
                El gradiente ∇f apunta en la dirección de **máximo crecimiento** de la función.
                Su magnitud indica la **tasa de cambio** en esa dirección.
                """)
            
            # VISUALIZACIÓN MEJORADA ESTILO "EL GRADIENTE"
            st.markdown("### 🎨 EL GRADIENTE - Visualización")
            
            tab1, tab2, tab3 = st.tabs(["🏔️ Superficie 3D", "🧭 Campo Vectorial 2D", "📊 Ambas Vistas"])
            
            with tab1:
                # SUPERFICIE 3D COLORIDA
                try:
                    x_surf = np.linspace(-3, 3, 60)
                    y_surf = np.linspace(-3, 3, 60)
                    X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
                    Z_surf = func_num(X_surf, Y_surf)
                    
                    fig_3d = go.Figure(data=[go.Surface(
                        x=X_surf, y=Y_surf, z=Z_surf,
                        colorscale='Rainbow',  # Colores vibrantes como en la imagen
                        showscale=True,
                        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5),
                        lightposition=dict(x=100, y=100, z=1000)
                    )])
                    
                    # Agregar curvas de nivel en la base
                    contour_3d = go.Contour(
                        x=x_surf, y=y_surf, z=Z_surf,
                        colorscale='Rainbow',
                        showscale=False,
                        contours=dict(start=np.min(Z_surf), end=np.max(Z_surf), size=(np.max(Z_surf)-np.min(Z_surf))/15),
                        opacity=0.3
                    )
                    
                    fig_3d.update_layout(
                        title=dict(text="🏔️ Superficie de la Función f(x,y)", font=dict(size=20)),
                        scene=dict(
                            xaxis_title="x",
                            yaxis_title="y",
                            zaxis_title="f(x,y)",
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                            bgcolor='rgb(20, 20, 30)'
                        ),
                        width=700,
                        height=600,
                        paper_bgcolor='rgb(20, 20, 30)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error en superficie 3D: {e}")
            
            with tab2:
                # CAMPO VECTORIAL 2D CON FLECHAS DE COLORES (OPTIMIZADO)
                try:
                    # Grilla REDUCIDA para el campo vectorial (más rápido)
                    density = 12  # Reducido de 20 a 12 para velocidad
                    x_vec = np.linspace(-3, 3, density)
                    y_vec = np.linspace(-3, 3, density)
                    X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
                    
                    # Calcular componentes del gradiente
                    fx_num = lambdify([x, y], fx, 'numpy')
                    fy_num = lambdify([x, y], fy, 'numpy')
                    
                    U = fx_num(X_vec, Y_vec)
                    V = fy_num(X_vec, Y_vec)
                    
                    # Normalizar para mejor visualización
                    magnitude = np.sqrt(U**2 + V**2)
                    magnitude_safe = np.where(magnitude == 0, 1, magnitude)
                    
                    # Crear figura
                    fig_2d = go.Figure()
                    
                    # Agregar curvas de nivel de fondo (RESOLUCIÓN REDUCIDA)
                    x_contour = np.linspace(-3, 3, 60)  # Reducido de 100 a 60
                    y_contour = np.linspace(-3, 3, 60)
                    X_contour, Y_contour = np.meshgrid(x_contour, y_contour)
                    Z_contour = func_num(X_contour, Y_contour)
                    
                    fig_2d.add_trace(go.Contour(
                        x=x_contour, y=y_contour, z=Z_contour,
                        colorscale='Rainbow',
                        showscale=False,
                        opacity=0.3,
                        contours=dict(showlabels=False),
                        line=dict(width=1)
                    ))
                    
                    # Preparar datos para flechas (MÉTODO OPTIMIZADO - una sola traza)
                    max_mag = np.max(magnitude)
                    
                    # Listas para almacenar todas las flechas
                    x_arrows = []
                    y_arrows = []
                    colors = []
                    
                    for i in range(len(x_vec)):
                        for j in range(len(y_vec)):
                            if magnitude[j, i] > 0.01:  # Evitar vectores muy pequeños
                                # Escalar la longitud de las flechas
                                scale = 0.2
                                
                                # Color basado en la magnitud
                                color_val = magnitude[j, i]
                                
                                # Escala de colores simplificada
                                if color_val < max_mag * 0.2:
                                    color = 'rgb(0, 100, 255)'  # Azul
                                elif color_val < max_mag * 0.4:
                                    color = 'rgb(0, 200, 150)'  # Verde-azul
                                elif color_val < max_mag * 0.6:
                                    color = 'rgb(200, 200, 0)'  # Amarillo
                                elif color_val < max_mag * 0.8:
                                    color = 'rgb(255, 150, 0)'  # Naranja
                                else:
                                    color = 'rgb(255, 50, 0)'   # Rojo
                                
                                # MÉTODO RÁPIDO: Usar anotaciones solo para flechas visibles
                                fig_2d.add_annotation(
                                    x=X_vec[j, i], 
                                    y=Y_vec[j, i],
                                    ax=X_vec[j, i] + U[j, i] * scale / magnitude_safe[j, i],
                                    ay=Y_vec[j, i] + V[j, i] * scale / magnitude_safe[j, i],
                                    xref='x', yref='y',
                                    axref='x', ayref='y',
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    arrowcolor=color,
                                    opacity=0.8
                                )
                    
                    # Marcar el punto específico
                    fig_2d.add_trace(go.Scatter(
                        x=[px], y=[py],
                        mode='markers',
                        marker=dict(size=15, color='white', symbol='star', 
                                  line=dict(color='red', width=2)),
                        name=f'Punto ({px}, {py})',
                        showlegend=True
                    ))
                    
                    # Agregar ejes en el centro
                    fig_2d.add_hline(y=0, line=dict(color='white', width=1, dash='dash'), opacity=0.5)
                    fig_2d.add_vline(x=0, line=dict(color='white', width=1, dash='dash'), opacity=0.5)
                    
                    fig_2d.update_layout(
                        title=dict(text="🧭 Campo Vectorial del Gradiente ∇f", font=dict(size=20)),
                        xaxis=dict(
                            title="x",
                            range=[-3, 3],
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.2)',
                            zeroline=True
                        ),
                        yaxis=dict(
                            title="y",
                            range=[-3, 3],
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.2)',
                            scaleanchor="x",
                            scaleratio=1,
                            zeroline=True
                        ),
                        width=700,
                        height=700,
                        paper_bgcolor='rgb(20, 20, 30)',
                        plot_bgcolor='rgb(20, 20, 30)',
                        font=dict(color='white'),
                        showlegend=True
                    )
                    
                    # Mostrar con mensaje de carga
                    with st.spinner('Generando campo vectorial...'):
                        st.plotly_chart(fig_2d, use_container_width=True)
                    
                    st.info("""
                    🎨 **Código de colores del gradiente:**
                    - 🔵 Azul: Cambio suave (magnitud baja)
                    - 🟢 Verde: Cambio moderado
                    - 🟡 Amarillo: Cambio significativo
                    - 🟠 Naranja: Cambio fuerte
                    - 🔴 Rojo: Cambio máximo (magnitud alta)
                    """)
                    
                except Exception as e:
                    st.error(f"Error en campo vectorial: {e}")
            
            with tab3:
                # VISTA COMBINADA (OPTIMIZADA)
                st.markdown("### 📐 Vista Completa: EL GRADIENTE")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    # Superficie 3D compacta (OPTIMIZADA)
                    try:
                        x_surf = np.linspace(-3, 3, 40)  # Reducido de 50 a 40
                        y_surf = np.linspace(-3, 3, 40)
                        X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
                        Z_surf = func_num(X_surf, Y_surf)
                        
                        fig_3d_small = go.Figure(data=[go.Surface(
                            x=X_surf, y=Y_surf, z=Z_surf,
                            colorscale='Rainbow',
                            showscale=False
                        )])
                        
                        fig_3d_small.update_layout(
                            title="Superficie f(x,y)",
                            scene=dict(
                                xaxis_title="x", yaxis_title="y", zaxis_title="f",
                                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                                bgcolor='rgb(20, 20, 30)'
                            ),
                            width=400, height=400,
                            paper_bgcolor='rgb(20, 20, 30)',
                            font=dict(color='white', size=10),
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        
                        with st.spinner('Generando superficie...'):
                            st.plotly_chart(fig_3d_small, use_container_width=True)
                    except:
                        st.error("Error al generar superficie 3D")
                
                with col_right:
                    # Campo vectorial compacto (MUY OPTIMIZADO)
                    try:
                        density = 10  # Reducido de 15 a 10 para mayor velocidad
                        x_vec = np.linspace(-3, 3, density)
                        y_vec = np.linspace(-3, 3, density)
                        X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
                        
                        fx_num = lambdify([x, y], fx, 'numpy')
                        fy_num = lambdify([x, y], fy, 'numpy')
                        
                        U = fx_num(X_vec, Y_vec)
                        V = fy_num(X_vec, Y_vec)
                        magnitude = np.sqrt(U**2 + V**2)
                        
                        fig_2d_small = go.Figure()
                        
                        # Fondo (RESOLUCIÓN REDUCIDA)
                        x_cont = np.linspace(-3, 3, 50)  # Reducido de 80 a 50
                        y_cont = np.linspace(-3, 3, 50)
                        X_cont, Y_cont = np.meshgrid(x_cont, y_cont)
                        Z_cont = func_num(X_cont, Y_cont)
                        
                        fig_2d_small.add_trace(go.Contour(
                            x=x_cont, y=y_cont, z=Z_cont,
                            colorscale='Rainbow', showscale=False,
                            opacity=0.3, contours=dict(showlabels=False)
                        ))
                        
                        # Vectores (menos densos)
                        max_mag = np.max(magnitude)
                        for i in range(len(x_vec)):
                            for j in range(len(y_vec)):
                                if magnitude[j, i] > 0.01:
                                    color_val = magnitude[j, i] / max_mag
                                    if color_val < 0.3:
                                        color = 'cyan'
                                    elif color_val < 0.6:
                                        color = 'yellow'
                                    else:
                                        color = 'red'
                                    
                                    scale = 0.15
                                    mag_safe = magnitude[j, i] if magnitude[j, i] > 0 else 1
                                    
                                    fig_2d_small.add_annotation(
                                        x=X_vec[j, i], y=Y_vec[j, i],
                                        ax=X_vec[j, i] + U[j, i] * scale / mag_safe,
                                        ay=Y_vec[j, i] + V[j, i] * scale / mag_safe,
                                        arrowhead=2, arrowsize=0.8, arrowwidth=1.5,
                                        arrowcolor=color, opacity=0.7
                                    )
                        
                        fig_2d_small.add_hline(y=0, line=dict(color='white', width=1), opacity=0.3)
                        fig_2d_small.add_vline(x=0, line=dict(color='white', width=1), opacity=0.3)
                        
                        fig_2d_small.update_layout(
                            title="Campo ∇f(x,y)",
                            xaxis=dict(title="x", range=[-3, 3], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                            yaxis=dict(title="y", range=[-3, 3], showgrid=True, gridcolor='rgba(255,255,255,0.1)', scaleanchor="x"),
                            width=400, height=400,
                            paper_bgcolor='rgb(20, 20, 30)',
                            plot_bgcolor='rgb(20, 20, 30)',
                            font=dict(color='white', size=10),
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        
                        with st.spinner('Generando campo vectorial...'):
                            st.plotly_chart(fig_2d_small, use_container_width=True)
                    except:
                        st.error("Error al generar campo vectorial")

# MÓDULO OPTIMIZACIÓN
elif modulo == "🎯 Optimización con Restricciones":
    st.markdown('<h2 class="sub-header">🎯 Optimización con Multiplicadores de Lagrange</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Configuración del Problema")
        func_input = st.text_input("Función objetivo f(x,y):", value="x**2 + y**2", help="Función a optimizar")
        constraint_input = st.text_input("Restricción g(x,y) = 0:", value="x + y - 1", help="Ejemplo: x + y - 1")
        
        opt_type = st.radio("Tipo de optimización:", ["Minimizar", "Maximizar"])
    
    if func_input and constraint_input:
        x, y, lam = symbols('x y lambda')
        
        try:
            f_sym = sp.sympify(func_input.replace('^', '**'))
            g_sym = sp.sympify(constraint_input.replace('^', '**'))
            
            with col2:
                st.markdown("### 🧮 Solución Analítica")
                st.markdown(f"**Función objetivo:** f(x,y) = {f_sym}")
                st.markdown(f"**Restricción:** g(x,y) = {g_sym} = 0")
                
                # Sistema de Lagrange
                fx = diff(f_sym, x)
                fy = diff(f_sym, y)
                gx = diff(g_sym, x)
                gy = diff(g_sym, y)
                
                st.markdown("**Sistema de ecuaciones:**")
                st.latex(r"\nabla f = \lambda \nabla g")
                st.latex(r"g(x,y) = 0")
                
                # Resolver sistema
                try:
                    eq1 = fx - lam * gx
                    eq2 = fy - lam * gy
                    eq3 = g_sym
                    
                    solutions = solve([eq1, eq2, eq3], [x, y, lam])
                    
                    if solutions:
                        st.markdown("### 📍 Puntos Críticos:")
                        
                        if isinstance(solutions, list):
                            for i, sol in enumerate(solutions):
                                if len(sol) >= 2:
                                    x_val, y_val = float(sol[0]), float(sol[1])
                                    f_val = float(f_sym.subs([(x, x_val), (y, y_val)]))
                                    st.write(f"**Punto {i+1}:** ({x_val:.4f}, {y_val:.4f})")
                                    st.write(f"**Valor:** f = {f_val:.4f}")
                        else:
                            if len(solutions) >= 2:
                                x_val, y_val = float(solutions[x]), float(solutions[y])
                                f_val = float(f_sym.subs([(x, x_val), (y, y_val)]))
                                st.write(f"**Punto óptimo:** ({x_val:.4f}, {y_val:.4f})")
                                st.write(f"**Valor óptimo:** f = {f_val:.4f}")
                    
                except Exception as e:
                    st.warning(f"No se pudo resolver analíticamente: {e}")
                    st.info("Intenta con funciones más simples o usa métodos numéricos")
            
            # Visualización
            st.markdown("### 🎨 Visualización")
            
            try:
                f_num = lambdify([x, y], f_sym, 'numpy')
                g_num = lambdify([x, y], g_sym, 'numpy')
                
                x_range = np.linspace(-3, 3, 100)
                y_range = np.linspace(-3, 3, 100)
                X, Y = np.meshgrid(x_range, y_range)
                Z = f_num(X, Y)
                
                fig = go.Figure()
                
                # Curvas de nivel de la función objetivo
                fig.add_trace(go.Contour(
                    x=x_range, y=y_range, z=Z,
                    colorscale='Viridis', opacity=0.7,
                    contours=dict(showlabels=True),
                    name='Función objetivo'
                ))
                
                # Restricción
                G = g_num(X, Y)
                fig.add_trace(go.Contour(
                    x=x_range, y=y_range, z=G,
                    contours=dict(coloring='lines', showlabels=True, start=0, end=0, size=1),
                    line=dict(color='red', width=3),
                    name='Restricción g=0'
                ))
                
                # Puntos críticos si existen
                if 'solutions' in locals() and solutions:
                    if isinstance(solutions, list):
                        for sol in solutions:
                            if len(sol) >= 2:
                                fig.add_trace(go.Scatter(
                                    x=[float(sol[0])], y=[float(sol[1])],
                                    mode='markers', marker=dict(size=12, color='red', symbol='star'),
                                    name='Punto crítico'
                                ))
                
                fig.update_layout(
                    title="Optimización con Restricciones",
                    xaxis_title="X", yaxis_title="Y",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error en la visualización: {e}")
                
        except Exception as e:
            st.error(f"Error al procesar las funciones: {e}")

# MÓDULO INTEGRALES MÚLTIPLES
elif modulo == "∫ Integrales Múltiples":
    st.markdown('<h2 class="sub-header">∫∫ Integrales Múltiples</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Integrales Dobles", "Integrales Triples"])
    
    with tab1:
        st.markdown("### Integral Doble: ∬ f(x,y) dA")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            func_input = st.text_input("Función f(x,y):", value="x*y", help="Ejemplo: x*y")
            
            st.markdown("**Límites de integración:**")
            x_min_expr = st.text_input("x desde:", value="0")
            x_max_expr = st.text_input("x hasta:", value="1")
            y_min_expr = st.text_input("y desde:", value="0", help="Puede depender de x")
            y_max_expr = st.text_input("y hasta:", value="x", help="Puede depender de x")
            
            calc_button = st.button("Calcular Integral", type="primary")
        
        with col2:
            if func_input and calc_button:
                x, y = symbols('x y')
                
                try:
                    f_sym = sp.sympify(func_input.replace('^', '**'))
                    
                    x_min_sym = sp.sympify(x_min_expr.replace('^', '**'))
                    x_max_sym = sp.sympify(x_max_expr.replace('^', '**'))
                    y_min_sym = sp.sympify(y_min_expr.replace('^', '**'))
                    y_max_sym = sp.sympify(y_max_expr.replace('^', '**'))
                    
                    st.markdown("**Setup de la integral:**")
                    st.latex(f"\\int_{{{x_min_sym}}}^{{{x_max_sym}}} \\int_{{{y_min_sym}}}^{{{y_max_sym}}} {sp.latex(f_sym)} \\, dy \\, dx")
                    
                    # Calcular integral simbólica
                    with st.spinner("Calculando..."):
                        try:
                            # Integrar primero respecto a y
                            inner_integral = integrate(f_sym, (y, y_min_sym, y_max_sym))
                            st.write(f"**Después de integrar en y:** {inner_integral}")
                            
                            # Luego integrar respecto a x
                            result = integrate(inner_integral, (x, x_min_sym, x_max_sym))
                            st.write(f"**Resultado:** {result}")
                            
                            # Evaluación numérica
                            if result.is_number:
                                st.success(f"**Valor numérico:** {float(result):.6f}")
                            else:
                                try:
                                    numeric_result = float(result.evalf())
                                    st.success(f"**Valor numérico:** {numeric_result:.6f}")
                                except:
                                    st.info("Resultado simbólico obtenido")
                            
                        except Exception as e:
                            st.error(f"Error en el cálculo simbólico: {e}")
                            
                            # Intentar integración numérica
                            try:
                                f_num = lambdify([x, y], f_sym, 'numpy')
                                
                                def integrand(y_val, x_val):
                                    return f_num(x_val, y_val)
                                
                                # Para límites constantes solamente
                                if (x_min_sym.is_number and x_max_sym.is_number and 
                                    y_min_sym.is_number and y_max_sym.is_number):
                                    
                                    result_num, error = dblquad(
                                        integrand,
                                        float(x_min_sym), float(x_max_sym),
                                        lambda x: float(y_min_sym), lambda x: float(y_max_sym)
                                    )
                                    st.success(f"**Integración numérica:** {result_num:.6f} ± {error:.2e}")
                                else:
                                    st.warning("Integración numérica no disponible para límites variables")
                                    
                            except Exception as e2:
                                st.error(f"Error en integración numérica: {e2}")
                
                except Exception as e:
                    st.error(f"Error al procesar la función: {e}")
        
        # Visualización de la región de integración
        if func_input:
            st.markdown("### 🎨 Región de Integración")
            
            try:
                x_vals = np.linspace(0, 2, 100)
                y_vals = np.linspace(0, 2, 100)
                X, Y = np.meshgrid(x_vals, y_vals)
                
                f_num = lambdify([x, y], sp.sympify(func_input.replace('^', '**')), 'numpy')
                Z = f_num(X, Y)
                
                fig = go.Figure(data=[go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale='Plasma',
                    opacity=0.8
                )])
                
                fig.update_layout(
                    title="Función a Integrar",
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    width=800, height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error en visualización: {e}")
    
    with tab2:
        st.markdown("### Integral Triple: ∭ f(x,y,z) dV")
        
        func_input_3d = st.text_input("Función f(x,y,z):", value="x*y*z", help="Ejemplo: x*y*z")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Límites en x:**")
            x_min_3d = st.text_input("x desde:", value="0", key="x_min_3d")
            x_max_3d = st.text_input("x hasta:", value="1", key="x_max_3d")
        
        with col2:
            st.markdown("**Límites en y:**")
            y_min_3d = st.text_input("y desde:", value="0", key="y_min_3d")
            y_max_3d = st.text_input("y hasta:", value="1", key="y_max_3d")
        
        with col3:
            st.markdown("**Límites en z:**")
            z_min_3d = st.text_input("z desde:", value="0", key="z_min_3d")
            z_max_3d = st.text_input("z hasta:", value="1", key="z_max_3d")
        
        calc_button_3d = st.button("Calcular Integral Triple", type="primary", key="calc_3d")
        
        if func_input_3d and calc_button_3d:
            x, y, z = symbols('x y z')
            
            try:
                f_sym_3d = sp.sympify(func_input_3d.replace('^', '**'))
                
                x_min_sym_3d = sp.sympify(x_min_3d.replace('^', '**'))
                x_max_sym_3d = sp.sympify(x_max_3d.replace('^', '**'))
                y_min_sym_3d = sp.sympify(y_min_3d.replace('^', '**'))
                y_max_sym_3d = sp.sympify(y_max_3d.replace('^', '**'))
                z_min_sym_3d = sp.sympify(z_min_3d.replace('^', '**'))
                z_max_sym_3d = sp.sympify(z_max_3d.replace('^', '**'))
                
                st.markdown("**Setup de la integral:**")
                st.latex(f"\\int_{{{x_min_sym_3d}}}^{{{x_max_sym_3d}}} \\int_{{{y_min_sym_3d}}}^{{{y_max_sym_3d}}} \\int_{{{z_min_sym_3d}}}^{{{z_max_sym_3d}}} {sp.latex(f_sym_3d)} \\, dz \\, dy \\, dx")
                
                with st.spinner("Calculando integral triple..."):
                    try:
                        # Integración paso a paso
                        inner_z = integrate(f_sym_3d, (z, z_min_sym_3d, z_max_sym_3d))
                        st.write(f"**Después de integrar en z:** {inner_z}")
                        
                        inner_y = integrate(inner_z, (y, y_min_sym_3d, y_max_sym_3d))
                        st.write(f"**Después de integrar en y:** {inner_y}")
                        
                        result_3d = integrate(inner_y, (x, x_min_sym_3d, x_max_sym_3d))
                        st.write(f"**Resultado:** {result_3d}")
                        
                        if result_3d.is_number:
                            st.success(f"**Valor numérico:** {float(result_3d):.6f}")
                        else:
                            try:
                                numeric_result = float(result_3d.evalf())
                                st.success(f"**Valor numérico:** {numeric_result:.6f}")
                            except:
                                st.info("Resultado simbólico obtenido")
                        
                    except Exception as e:
                        st.error(f"Error en cálculo simbólico: {e}")
                        
                        # Integración numérica para límites constantes
                        try:
                            if all(sym.is_number for sym in [x_min_sym_3d, x_max_sym_3d, 
                                                           y_min_sym_3d, y_max_sym_3d, 
                                                           z_min_sym_3d, z_max_sym_3d]):
                                
                                f_num_3d = lambdify([x, y, z], f_sym_3d, 'numpy')
                                
                                def integrand_3d(z_val, y_val, x_val):
                                    return f_num_3d(x_val, y_val, z_val)
                                
                                result_num_3d, error_3d = tplquad(
                                    integrand_3d,
                                    float(x_min_sym_3d), float(x_max_sym_3d),
                                    lambda x: float(y_min_sym_3d), lambda x: float(y_max_sym_3d),
                                    lambda x, y: float(z_min_sym_3d), lambda x, y: float(z_max_sym_3d)
                                )
                                st.success(f"**Integración numérica:** {result_num_3d:.6f} ± {error_3d:.2e}")
                            else:
                                st.warning("Integración numérica no disponible para límites variables")
                        
                        except Exception as e2:
                            st.error(f"Error en integración numérica: {e2}")
            
            except Exception as e:
                st.error(f"Error al procesar la función: {e}")

# MÓDULO AYUDA Y EJEMPLOS
elif modulo == "📖 Ayuda y Ejemplos":
    st.markdown('<h2 class="sub-header">📖 Ayuda y Ejemplos</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📝 Sintaxis", "🔗 Ejemplos", "⚡ Casos de Uso"])
    
    with tab1:
        st.markdown("""
        ## 📝 Guía de Sintaxis
        
        ### Operadores Matemáticos
        - **Suma:** `+`
        - **Resta:** `-`
        - **Multiplicación:** `*`
        - **División:** `/`
        - **Potencia:** `**` (ejemplo: `x**2` para x²)
        
        ### Funciones Disponibles
        - **Trigonométricas:** `sin(x)`, `cos(x)`, `tan(x)`
        - **Exponencial:** `exp(x)` (e^x)
        - **Logaritmo:** `log(x)` (ln x)
        - **Raíz cuadrada:** `sqrt(x)`
        - **Valor absoluto:** `Abs(x)`
        
        ### Constantes
        - **Pi:** `pi`
        - **Euler:** `E`
        
        ### Variables
        - Para funciones 2D: usar `x` e `y`
        - Para funciones 3D: usar `x`, `y` y `z`
        """)
    
    with tab2:
        st.markdown("""
        ## 🔗 Ejemplos por Módulo
        
        ### 📊 Visualización
        ```python
        # Paraboloide
        x**2 + y**2
        
        # Silla de montar
        x**2 - y**2
        
        # Función sinusoidal
        sin(x) * cos(y)
        
        # Gaussiana 2D
        exp(-(x**2 + y**2))
        ```
        
        ### 📐 Derivadas Parciales
        ```python
        # Para análisis de gradiente
        x**3 + y**3 - 3*x*y
        
        # Función con máximo/mínimo
        -x**2 - y**2 + 4*x + 6*y
        ```
        
        ### 🎯 Optimización
        ```python
        # Función objetivo
        x**2 + y**2
        
        # Restricción (g = 0)
        x + y - 1
        ```
        
        ### ∫ Integrales
        ```python
        # Integrales dobles
        x*y           # Límites: x=0 a 1, y=0 a x
        x**2 + y**2   # Límites rectangulares
        
        # Integrales triples
        x*y*z         # Para volúmenes
        1             # Para calcular volumen de región
        ```
        """)
    
    with tab3:
        st.markdown("""
        ## ⚡ Casos de Uso Prácticos
        
        ### 🏗️ Ingeniería
        **Distribución de temperatura en una placa:**
        ```
        Función: 100 - x**2 - y**2
        Dominio: Circular |x²+y²| ≤ 25
        ```
        
        **Optimización de costos:**
        ```
        Minimizar: x**2 + y**2 (costo)
        Restricción: x + y - 10 (producción)
        ```
        
        ### 🔬 Física
        **Campo gravitacional:**
        ```
        Función: -1/sqrt(x**2 + y**2 + 1)
        Gradiente muestra dirección de fuerza
        ```
        
        **Volumen bajo superficie:**
        ```
        Integral: ∬ f(x,y) dA
        Representa volumen sólido
        ```
        
        ### 📈 Economía
        **Función de utilidad:**
        ```
        U(x,y) = x**0.5 * y**0.5
        Optimizar con restricción presupuestaria
        ```
        
        **Centro de masa:**
        ```
        Densidad: ρ(x,y) = x + y
        Calcular: ∬ ρ(x,y) dA
        ```
        """)
    
    # Sección de tips
    st.markdown("""
    ## 💡 Tips y Trucos
    
    - **Visualización lenta?** Reduce el rango de x,y (por ejemplo: -2 a 2)
    - **Error de sintaxis?** Usa `**` en lugar de `^` para potencias
    - **Función compleja?** Comienza con ejemplos simples como `x**2 + y**2`
    - **Dominio extraño?** Revisa funciones con `sqrt`, `log` o divisiones
    - **Optimización no converge?** Usa funciones suaves y restricciones simples
    
    ## 🆘 Resolución de Problemas
    
    | Problema | Solución |
    |----------|----------|
    | "Error al interpretar función" | Revisa sintaxis, usa `**` para potencias |
    | "No se puede graficar" | Verifica que la función esté bien definida |
    | "División por cero" | Ajusta el dominio para evitar singularidades |
    | "Integración falla" | Usa límites simples o funciones más básicas |
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🧮 Calculadora de Cálculo Multivariable | Desarrollado con Streamlit</p>
    <p>Para soporte técnico y mejoras, consulta la documentación del proyecto</p>
</div>
""", unsafe_allow_html=True)