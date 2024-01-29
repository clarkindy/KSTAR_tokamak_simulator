#!/usr/bin/env python

import os
import sys
import typing
from collections import deque, namedtuple
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
import plotly.subplots
import streamlit as st
from PIL import Image
from scipy import interpolate

from common.model_structure import *
from common.utils import point_in_polygon
from common.wall import *

# Setting
base_path = os.path.abspath(os.path.dirname(sys.argv[0]))
background_path = base_path + "/images/insideKSTAR.jpg"
lstm_model_path = base_path + "/weights/lstm/v220505/"
nn_model_path = base_path + "/weights/nn/"
bpw_model_path = base_path + "/weights/bpw/v220505/"
k2rz_model_path = base_path + "/weights/k2rz/"
MAX_MODELS = 10
MAX_SHAPE_MODELS = 1
decimals = np.log10(200)
DPI = 1
PLOT_LENGTH = 40
YEAR_IN = 2021
EC_FREQ = 105.0e9
_COLOR_TRANSPARENT = "rgba(0,0,0,0)"

# Inputs
input_names = [
    "ip",
    "bt",
    "fgw",
    "pnb1a",
    "pnb1b",
    "pnb1c",
    "pec2",
    "pec3",
    "zec2",
    "zec3",
    "rin",
    "rout",
    "k",
    "du",
    "dl",
]

LstmInputRow = namedtuple(
    "LstmInputRow",
    (
        "betan",
        "q95",
        "q0",
        "li",
        "ip",
        "bt",
        "fgw",
        "k",
        "du",
        "dl",
        "rin",
        "rout",
        "pnb1a",
        "pnb1b",
        "pnb1c",
        "pec",
        "div",
        "yr",
    ),
)


input_params = ['Ip [MA]','Bt [T]','GW.frac. [-]',\
                'Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]',\
                'Pec2 [MW]','Pec3 [MW]','Zec2 [cm]','Zec3 [cm]',\
                'In.Mid. [m]','Out.Mid. [m]','Elon. [-]','Up.Tri. [-]','Lo.Tri [-]']
input_mins = [0.3,1.5,0.2, 0.0, 0.0, 0.0, 0.0,0.0,-10.0,-10.0, 1.265,2.18,1.6,0.1,0.5 ]
input_maxs = [0.8,2.7,0.6, 1.75,1.75,1.5, 0.8,0.8, 10.0, 10.0, 1.36, 2.29,2.0,0.5,0.9 ]
input_init = [0.5,1.8,0.4, 1.5, 0.0, 0.0, 0.0,0.0,  0.0,  0.0, 1.34, 2.22,1.7,0.3,0.75]

# Outputs
output_params0 = ['betan','q95','q0','li']
output_params1 = ['betap','wmhd']
output_params2 = ['betan','betap','h89','h98','q95','q0','li','wmhd']

class Session:
    n_models: int
    kstar_nn: kstar_nn
    kstar_lstm: kstar_v220505
    k2rz: k2rz
    bpw_nn: bpw_nn
    ip: float  # Plasma current [MA]
    bt: float  # Toroidal magnetic field [T]
    fgw: float  # Greenwald density fraction
    pnb1a: float  # NB1A power [MW]
    pnb1b: float  # NB1B power [MW]
    pnb1c: float  # NB1C power [MW]
    pec2: float  # EC2 power [MW]
    pec3: float  # EC3 power [MW]
    zec2: float  # EC2 angle (Z @ R=R0) [cm]
    zec3: float  # EC3 angle (Z @ R=R0) [cm]
    rin: float  # Major radius @ inboard midplane [m]
    rout: float  # Major radius @ outboard midplane [m]
    k: float  # Elongation
    du: float  # Upper triangularity
    dl: float  # Lower triangularity
    betan: deque[float]  # Normalized beta
    betap: deque[float]  # Poloidal beta
    h89: deque[float]  # Confinement enhancement factor ITER89-P
    h98: deque[float]  # Confinement enhancement factor IPB98(y,2)
    q95: deque[float]  # q @ 95% poloidal flux surface
    q0: deque[float]  # q @ Magnetic axis
    li: deque[float]  # Internal inductance (normalized)
    wmhd: deque[float]  # Plasma energy [MJ]
    fgw_history: deque[float]
    lstm_in: deque[LstmInputRow]
    predict: bool
    dump_outputs: bool
    initialized: bool


def initialize_session(session: Session):
    n_models = MAX_MODELS
    session.n_models = n_models
    # Load models
    session.kstar_nn = kstar_nn(model_path=nn_model_path, n_models=1)
    session.kstar_lstm = kstar_v220505(model_path=lstm_model_path, n_models=n_models)
    session.k2rz = k2rz(model_path=k2rz_model_path, n_models=MAX_SHAPE_MODELS)
    session.bpw_nn = tf_dense_model(
        model_path=bpw_model_path,
        n_models=n_models,
        ymean=[1.3630552066021155, 251779.19861710534],
        ystd=[0.6252123013157276, 123097.77805034176],
    )
    # Initialize input parameters
    for name, init in zip(input_names, input_init):
        setattr(session, name, init)
    session.fgw_history = deque([session.fgw], maxlen=PLOT_LENGTH)
    # Initialize output parameters
    for name in output_params2:
        setattr(session, name, deque(maxlen=PLOT_LENGTH))
    # Initialize LSTM input
    session.lstm_in = deque(maxlen=10)
    predict0d(session, steady=True)
    session.predict = False
    session.dump_outputs = False
    session.initialized = True


def render_kstar():
    st.markdown(
        """\
        <style>
          /* Wider main section */
          .main > .block-container {
            max-width: 96rem;
          }

          /* Top layout */
          .main
            > .block-container
            > :first-child
            > :first-child
            > :first-child
            > :nth-child(3) {
            align-items: end;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("KSTAR-NN Simulator")
    session = typing.cast(Session, st.session_state)

    # Initial condition
    if not getattr(st.session_state, "initialized", False):
        initialize_session(session)

    # Top layout
    top1, top2, top3, top4 = st.columns([1, 1, 1, 1])
    with top1:
        st.number_input(
            "\\# of models",
            min_value=1,
            max_value=MAX_MODELS,
            key="n_models",
            on_change=reset_model_number,
        )
    with top2:
        st.button("Shuffle models", on_click=shuffle_models)
    with top3:
        is_running = st.toggle("Run")
    with top4:
        over_plot = st.checkbox("Overlap device")

    if is_running and session.predict:
        predict0d(session, steady=False)
        session.predict = False

    # Middle layout
    # middle1, middle2, middle3 = st.columns([1, 1, 1])
    middle2, middle3 = st.columns([1, 1])
    # with middle1:
    with st.sidebar:
        st.markdown("#### Input parameters")
        # with st.container(height=500):
        #     render_input_box()
        render_input_box()
    with middle2:
        st.markdown("#### 2D poloidal view")
        fig = go.Figure()
        if over_plot:
            plot_background(fig)
        plot_wall(fig, Rwalls, Zwalls)
        rbdry, zbdry = predict_boundary(session)
        plot_plasma_boundary(fig, rbdry, zbdry)
        plot_x_points(fig, rbdry, zbdry)
        plot_nbi(fig, session.pnb1a, session.pnb1b, session.pnb1c)
        plot_ec(fig, session.pec2, session.pec3, session.zec2, session.zec3, session.bt)
        plot_head_loads(fig, rbdry, zbdry)
        fig.update_layout(dict(height=500, xaxis_title="R [m]", yaxis_title="Z [m]"))
        fig.update_xaxes(range=[0.8, 2.5])
        fig.update_yaxes(range=[-1.55, 1.55])
        st.plotly_chart(fig, use_container_width=True)
    with middle3:
        st.markdown("#### 0D evolution")
        if session.dump_outputs:
            render_evolution_df(session)
        else:
            render_evolution_plot(session)

    # Bottom layout
    # bottom1, bottom2, bottom3 = st.columns([1, 1, 1])
    bottom1, bottom2, bottom3 = st.columns([1, 1, 2])
    with bottom1:
        st.button(
            "▶▶ 1s ▶▶",
            use_container_width=True,
            on_click=relax_run,
            kwargs=dict(seconds=1),
        )
    with bottom2:
        st.button(
            "▶▶ 2s ▶▶",
            use_container_width=True,
            on_click=relax_run,
            kwargs=dict(seconds=2),
        )
    with bottom3:
        st.button(
            "Toggle display (Plot ↔ Table)",
            use_container_width=True,
            on_click=toggle_evolution_mode,
        )


def reset_model_number():
    session = typing.cast(Session, st.session_state)
    session.kstar_lstm.nmodels = session.n_models
    session.bpw_nn.nmodels = session.n_models


def predict_next():
    session = typing.cast(Session, st.session_state)
    session.predict = True


def render_input_box():
    for name, param, min_, max_ in zip(
        input_names, input_params, input_mins, input_maxs
    ):
        st.slider(
            label=param,
            min_value=min_,
            max_value=max_,
            step=(max_ - min_) / 200,
            key=name,
            on_change=predict_next,
        )


def plot_wall(fig: go.Figure, rwall: np.ndarray, zwall: np.ndarray, **kwargs):
    fig.add_trace(
        go.Scatter(x=rwall, y=zwall, name="Wall", line=dict(color="black")),
        **kwargs,
    )


def plot_plasma_boundary(
    fig: go.Figure, rbdry: np.ndarray, zbdry: np.ndarray, **kwargs
):
    color_base = plotly.colors.hex_to_rgb("#4C72B0")
    alpha = 0.2
    color = f"rgba({','.join(map(str, color_base))},{alpha})"
    fig.add_trace(
        go.Scatter(
            x=rbdry,
            y=zbdry,
            name="LCFS",
            legendgroup="Plasma",
            legendgrouptitle=dict(text="Plasma"),
            line=dict(color=f"rgb({','.join(map(str, color_base))})"),
            fill="toself",
            fillcolor=color,
        ),
        **kwargs,
    )


def get_x_points(rbdry: np.ndarray, zbdry: np.ndarray):
    x1_index = np.argmin(zbdry)
    rx1 = rbdry[x1_index]
    zx1 = zbdry[x1_index]
    r = [rx1, rx1]
    z = [zx1, -zx1]
    return r, z, x1_index


def predict_boundary(session: Session):
    ip = session.ip
    bt = session.bt
    bp = session.betap[-1]
    rin = session.rin
    rout = session.rout
    k = session.k
    du = session.du
    dl = session.dl

    session.k2rz.set_inputs(ip, bt, bp, rin, rout, k, du, dl)
    rbdry, zbdry = session.k2rz.predict(post=True)
    return rbdry, zbdry


def plot_x_points(fig: go.Figure, rbdry: np.ndarray, zbdry: np.ndarray, **kwargs):
    r, z, _ = get_x_points(rbdry, zbdry)
    fig.add_trace(
        go.Scatter(
            x=r,
            y=z,
            name="X-Points",
            legendgroup="Plasma",
            legendgrouptitle=dict(text="Plasma"),
            marker=dict(color="white", symbol="x", size=12),
            line=dict(color=_COLOR_TRANSPARENT),
        ),
        **kwargs,
    )


def plot_background(fig: go.Figure, **kwargs):
    fig.add_layout_image(
        source=Image.open(background_path),
        xref="x",
        yref="y",
        x=-1.6,
        y=1.35,
        sizex=4.05,
        sizey=2.85,
        sizing="stretch",
        layer="below",
        **kwargs,
    )


def plot_head_loads(
    fig: go.Figure,
    rbdry: np.ndarray,
    zbdry: np.ndarray,
    n: int = 10,
    both_side: bool = True,
):
    color_base = plotly.colors.hex_to_rgb("#C44E52")
    alphas = (0.2, 1.0)
    colors = tuple(
        f"rgba({','.join(map(str, color_base))},{alpha})" for alpha in alphas
    )

    if (Rwalls[0], Zwalls[0]) == (Rwalls[-1], Zwalls[-1]):
        r = np.array([Rwalls[:-1], Rwalls[1:]])
        z = np.array([Zwalls[:-1], Zwalls[1:]])
    else:
        r = np.array([Rwalls, np.roll(Rwalls, -1)])
        z = np.array([Zwalls, np.roll(Zwalls, -1)])

    rx, zx, index = get_x_points(rbdry, zbdry)
    r_wall = np.min(Rwalls) + 1.0e-4
    z_wall = np.min(Zwalls) + 1.0e-4

    rx1_plot_: list[Sequence[float]] = []
    zx1_plot_: list[Sequence[float]] = []
    rx2_plot_: list[Sequence[float]] = []
    zx2_plot_: list[Sequence[float]] = []

    # kinds = ["linear", "cubic"]
    kinds = ["linear", "quadratic"]
    for i0, i1 in ((1, 6), (0, 6)):
        for kind in kinds:
            rsol1_sample = rbdry[index - i0 : index - i1 : -1]
            zsol1_sample = zbdry[index - i0 : index - i1 : -1]
            f1 = interpolate.interp1d(
                rsol1_sample, zsol1_sample, kind=kind, fill_value="extrapolate"
            )
            rsol1 = np.linspace(rx[0], r_wall, n)
            zsol1 = f1(rsol1)
            is_inside1 = point_in_polygon(rsol1, zsol1, r, z)
            rsol1_in = rsol1[is_inside1]
            zsol1_in = zsol1[is_inside1]

            rsol2_sample = rbdry[index + i0 : index + i1]
            zsol2_sample = zbdry[index + i0 : index + i1]
            f2 = interpolate.interp1d(
                zsol2_sample, rsol2_sample, kind=kind, fill_value="extrapolate"
            )
            zsol2 = np.linspace(zx[0], z_wall, n)
            rsol2 = f2(zsol2)
            is_inside2 = point_in_polygon(rsol2, zsol2, r, z)
            rsol2_in = rsol2[is_inside2]
            zsol2_in = zsol2[is_inside2]

            if not np.all(zsol1 > zbdry[index + 1]):  # ???
                rx1_plot_.append(rsol1_in)
                zx1_plot_.append(zsol1_in)
            rx1_plot_.append(rsol2_in)
            zx1_plot_.append(zsol2_in)
            if both_side:
                rx2_plot_.append(rsol1_in)
                zx2_plot_.append(-zsol1_in)
                rx2_plot_.append(rsol2_in)
                zx2_plot_.append(-zsol2_in)

    rx1_plot: list[Optional[float]] = []
    zx1_plot: list[Optional[float]] = []
    rx2_plot: list[Optional[float]] = []
    zx2_plot: list[Optional[float]] = []
    for from_, to in [
        (rx1_plot_, rx1_plot),
        (zx1_plot_, zx1_plot),
        (rx2_plot_, rx2_plot),
        (zx2_plot_, zx2_plot),
    ]:
        to.extend(x for array in from_ for seq in (array, (None,)) for x in seq)
    fig.add_trace(
        go.Scatter(
            mode="lines",
            x=rx1_plot,
            y=zx1_plot,
            name="From lower X-point",
            legendgroup="Heat load",
            legendgrouptitle=dict(text="Heat load"),
            line=dict(color=colors[1]),
        )
    )
    if both_side:
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=rx2_plot,
                y=zx2_plot,
                name="From upper X-point",
                legendgroup="Heat load",
                legendgrouptitle=dict(text="Heat load"),
                line=dict(color=colors[0]),
            )
        )


def plot_nbi(fig: go.Figure, pnb1a: float, pnb1b: float, pnb1c: float, **kwargs):
    color_base = plotly.colors.hex_to_rgb("#55A868")
    alphas = (0.3, 0.9)
    colors = tuple(
        f"rgba({','.join(map(str, color_base))},{alpha})" for alpha in alphas
    )
    nb_colors = [colors[pnb > 0.5] for pnb in (pnb1a, pnb1b, pnb1c)]
    w = 0.13
    h = 0.45
    fig.add_traces(
        [
            go.Scatter(
                x=[rt - w / 2, rt + w / 2, rt + w / 2, rt - w / 2],
                y=[-h / 2, -h / 2, h / 2, h / 2],
                name=name,
                legendgroup="NBI",
                legendgrouptitle=dict(text="NBI"),
                line=dict(color=_COLOR_TRANSPARENT),
                fill="toself",
                fillcolor=color,
            )
            for name, rt, color in zip(
                ["NB1A", "NB1B", "NB1C"], [1.486, 1.720, 1.245], nb_colors
            )
        ],
        **kwargs,
    )


def plot_ec(
    fig: go.Figure,
    pec2: float,
    pec3: float,
    zec2: float,
    zec3: float,
    bt: float,
    **kwargs,
):
    # Get resonance location
    for ns in [1, 2, 3]:
        rs = 1.60219e-19 * 1.8 * bt / (2.0 * np.pi * 9.10938e-31 * EC_FREQ) * ns
        if min(Rwalls) < rs < max(Rwalls):
            break

    color_base = plotly.colors.hex_to_rgb("#CCB974")
    alphas = (0.3, 0.9)
    colors = tuple(
        f"rgba({','.join(map(str, color_base))},{alpha})" for alpha in alphas
    )
    ec_colors = [colors[pec > 0.2] for pec in (pec2, pec3)]
    rpos = [2.449, 2.451]
    zpos = [0.35, -0.35]
    zres = [
        z + (zec / 100 - z) * (rs - z) / (1.8 - z)
        for z, z, zec in zip(rpos, zpos, (zec2, zec3))
    ]
    dz = 0.05
    fig.add_traces(
        [
            go.Scatter(
                x=[rpos_, rs, rs],
                y=[zpos_, zres_ - dz, zres_ + dz],
                name=name,
                legendgroup="EC",
                legendgrouptitle=dict(text="EC"),
                line=dict(color=_COLOR_TRANSPARENT),
                fill="toself",
                fillcolor=color,
            )
            for name, rpos_, zpos_, zres_, color in zip(
                ["EC2", "EC3"], rpos, zpos, zres, ec_colors
            )
        ],
        **kwargs,
    )


def get_radii(rin: float, rout: float):
    # Convert rin and rout to major and minor radii
    return 0.5 * (rin + rout), 0.5 * (rout - rin)


def new_lstm_input_row(session: Session):
    return LstmInputRow(
        session.betan[-1],
        session.q95[-1],
        session.q0[-1],
        session.li[-1],
        session.ip,
        session.bt,
        session.fgw,
        session.k,
        session.du,
        session.dl,
        session.rin,
        session.rout,
        session.pnb1a,
        session.pnb1b,
        session.pnb1c,
        session.pec2 + session.pec3,
        float(session.rin > 1.265 + 1.0e-4),
        float(YEAR_IN),
    )


def update_lstm_outputs(session: Session, y: np.ndarray):
    outputs = [session.betan, session.q95, session.q0, session.li]
    for output, y_ in zip(outputs, y):
        # if len(output) == 1:
        #     output[0] = y_
        output.append(y_)
    session.fgw_history.append(session.fgw)


def predict_kstar_nn(session: Session):
    idx_convert = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 2]
    x = np.array([*(session[input_names[i]] for i in idx_convert), 0])
    x[9], x[10] = get_radii(x[9], x[10])
    x[14] = x[14] > 1.265 + 1.0e-4
    x[-1] = YEAR_IN
    y = session.kstar_nn.predict(x)
    update_lstm_outputs(session, y)
    row = new_lstm_input_row(session)
    for _ in range(session.lstm_in.maxlen or 1):
        session.lstm_in.append(row)


def predict_kstar_lstm(session: Session):
    session.lstm_in.append(new_lstm_input_row(session))
    y = session.kstar_lstm.predict(np.array(session.lstm_in))
    update_lstm_outputs(session, y)


def predict_bpw_nn(session: Session):
    x = [
        session.betan[-1],
        session.ip,
        session.bt,
        session.rin,
        session.rout,
        session.k,
        session.du,
        session.dl,
    ]
    x[3], x[4] = get_radii(x[3], x[4])
    y = session.bpw_nn.predict(x)
    outputs = [session.betap, session.wmhd]
    for output, y_ in zip(outputs, y):
        # if len(output) == 1:
        #     output[0] = y_
        output.append(y_)


def get_h_factors(
    ip: float,
    bt: float,
    fgw: float,
    ptot: float,
    rin: float,
    rout: float,
    k: float,
    wmhd: float,
):
    r, a = get_radii(rin, rout)
    ne = 10 * fgw * (ip / (np.pi * a**2))  # Electron density [10^19 m^-3]
    m = 2.0  # Average mass of fuel atom [amu]

    x = np.log([ip, bt, ne, ptot, r, k, a / r, m])
    tau89 = 0.038 * np.exp(np.dot(x, [0.85, 0.2, 0.1, -0.5, 1.5, 0.5, 0.3, 0.5]))
    tau98 = 0.0562 * np.exp(
        np.dot(x, [0.93, 0.15, 0.41, -0.69, 1.97, 0.78, 0.58, 0.19])
    )
    h89 = 1.0e-6 * wmhd / ptot / tau89
    h98 = 1.0e-6 * wmhd / ptot / tau98
    return h89, h98


def update_h_factors(session: Session):
    h89, h98 = get_h_factors(
        session.ip,
        session.bt,
        session.fgw,
        max(
            sum(
                [
                    session.pnb1a,
                    session.pnb1b,
                    session.pnb1c,
                    session.pec2,
                    session.pec3,
                ]
            ),
            0.1,  # Not to diverge
        ),
        session.rin,
        session.rout,
        session.k,
        session.wmhd[-1],
    )
    session.h89.append(h89)
    session.h98.append(h98)


def predict0d(session: Session, steady: bool = False):
    if steady:
        predict_kstar_nn(session)
    else:
        predict_kstar_lstm(session)
    predict_bpw_nn(session)
    update_h_factors(session)


def shuffle_models():
    session = typing.cast(Session, st.session_state)
    np.random.shuffle(session.k2rz.models)
    np.random.shuffle(session.kstar_lstm.models)
    np.random.shuffle(session.bpw_nn.models)
    st.toast("Models shuffled!")


def relax_run(steps=None, seconds=None):
    session = typing.cast(Session, st.session_state)
    if steps is None and seconds is None:
        raise ValueError("Either steps or seconds should be given")
    if steps is None:
        steps = 10 * seconds
    for _ in range(steps - 1):
        predict0d(session, steady=False)


def toggle_evolution_mode():
    session = typing.cast(Session, st.session_state)
    session.dump_outputs = not session.dump_outputs


def render_evolution_df(session: Session):
    n = len(session.betan)
    time_ = np.linspace(-0.1 * (n - 1), 0, n)
    df = pd.DataFrame(
        {
            "Time [s]": time_,
            "βn": session.betan,
            "βp": session.betap,
            "H89": session.h89,
            "H98(y,2)": session.h98,
            "q95": session.q95,
            "q0": session.q0,
            "li": session.li,
            "Wmhd [100 kJ]": session.wmhd,
        }
    ).iloc[::-1]
    st.dataframe(df, use_container_width=True)


def render_evolution_plot(session: Session):
    time_ = np.linspace(-0.1 * (PLOT_LENGTH - 1), 0, PLOT_LENGTH)
    time_avail = time_[-len(session.betan) :]
    fig = plotly.subplots.make_subplots(
        rows=4,
        cols=1,
        shared_xaxes="columns",
    )

    fig.add_traces(
        [
            go.Scatter(
                mode="lines",
                x=time_avail,
                y=np.array(session.betan),
                name="βn",
                legendgroup="1",
            ),
            go.Scatter(
                mode="lines",
                x=time_avail,
                y=4 * np.array(session.li),
                name="4\u00d7li",
                legendgroup="1",
            ),
        ],
        rows=1,
        cols=1,
    )
    fig.add_traces(
        [
            go.Scatter(
                mode="lines",
                x=time_avail,
                y=1.0e-5 * np.array(session.wmhd),
                name="Wmhd [100 kJ]",
                legendgroup="2",
            ),
            go.Scatter(
                mode="lines",
                x=time_avail,
                y=np.array(session.h89),
                name="H89",
                legendgroup="2",
            ),
        ],
        rows=2,
        cols=1,
    )
    fig.add_traces(
        [
            go.Scatter(
                mode="lines",
                x=time_avail,
                y=np.array(session.fgw_history),
                name="ne/nGW",
                legendgroup="3",
            ),
            go.Scatter(
                mode="lines",
                x=time_avail,
                y=np.array(session.h98),
                name="H98(y,2)",
                legendgroup="3",
            ),
        ],
        rows=3,
        cols=1,
    )
    fig.add_traces(
        [
            go.Scatter(
                mode="lines",
                x=time_avail,
                y=np.array(session.q95),
                name="q95",
                legendgroup="4",
            ),
            go.Scatter(
                mode="lines",
                x=time_avail,
                y=np.array(session.q0),
                name="q0",
                legendgroup="4",
            ),
        ],
        rows=4,
        cols=1,
    )
    fig.update_layout(dict(height=500, legend_tracegroupgap=48))
    fig.update_xaxes(
        dict(range=(time_[0], time_[-1]), title="Relative time [s]"), row=4
    )
    st.plotly_chart(fig, theme=None, use_container_width=True)


if __name__ == "__main__":
    render_kstar()
