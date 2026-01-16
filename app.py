import dash
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import copy 


# ---------- HELPER FUNCTIONS ----------


def compute_profile_distance(row, global_mean):
    # Euclidean distance in feature space
    diffs = np.array([row[f] - global_mean[f] for f in FEATURES_PCA], dtype=float)
    return float(np.linalg.norm(diffs))


def compute_distance_between_rows(row, ref_row):
    diffs = np.array([row[f] - ref_row[f] for f in FEATURES_PCA], dtype=float)
    return float(np.linalg.norm(diffs))


def compute_feature_differences(selection_mean, global_mean):
    diffs = []
    for f in FEATURES_PCA:
        diffs.append(
            {
                "feature": f,
                "diff": float(selection_mean[f] - global_mean[f]),
                "abs_diff": float(abs(selection_mean[f] - global_mean[f])),
            }
        )
    # sort by absolute difference, descending
    diffs = sorted(diffs, key=lambda x: x["abs_diff"], reverse=True)
    return diffs


def empty_bar_figure():
    fig = go.Figure()
    fig.update_layout(
        title="Scores of Selected Universities",
        xaxis_title="Institution",
        yaxis_title="Score",
    )
    return fig


# ---------- LOAD DATA ----------


df = pd.read_csv("qs_2025_pca.csv")
df = df.reset_index(drop=False).rename(columns={"index": "row_id"})


df = df.replace("-", pd.NA)



FEATURES_PCA = [
    "Academic Reputation",
    "Employer Reputation",
    "Faculty Student",
    "Citations per Faculty",
    "International Faculty",
    "International Students",
    "International Research Network",
    "Employment Outcomes",
    "Sustainability",
]


NUMERIC_COLS = FEATURES_PCA + ["QS Overall Score"]
df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
GLOBAL_MEAN = df[FEATURES_PCA].mean()


META_COLS = [
    "Institution Name",
    "Location",
    "2025 Rank",
    "QS Overall Score",
]


# ---------- INITIAL FIGURES ----------
pca_fig = px.scatter(
    df,
    x="pca_1",
    y="pca_2",
    color="Location",
    hover_data=META_COLS,
    custom_data=["row_id"],
    size="Academic Reputation",
    size_max=15,
    title="PCA Projection of QS 2025 Universities",
)


pca_fig.update_layout(
    xaxis_title="PCA 1",
    yaxis_title="PCA 2",
    legend_title_text="Location",
)



def make_empty_radar():
    fig = go.Figure()
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Select universities to see their profile",
    )
    return fig



# ---------- APP SETUP ----------


app = Dash(__name__)


app.layout = html.Div(
    style={"padding": "20px", "fontFamily": "Arial, sans-serif"},
    
    children=[ 
        dcc.Store(id="selected-indices", data=[]), 
        html.H1(
            "QS World University Rankings 2025",
            style={"textAlign": "center"},
        ),


        html.Div(
            style={
                "display": "flex",
                "gap": "20px",
                "alignItems": "flex-start",
            },
            children=[
                html.Div(
                    style={"flex": "3"},
                    children=[
                        html.P(
                            "PCA map of universities based on QS indicators. "
                            "Use lasso/box selection on this plot to define groups; "
                            "the panels on the right summarize the selected group.",
                            style={"fontSize": "12px", "textAlign": "center", "marginTop": "4px"},
                        ),
                        dcc.Graph(
                            id="pca-scatter",
                            figure=pca_fig,
                            style={"height": "450px"},
                        ),
                        html.Div(
                            html.Button("Clear selection", id="clear-selection", n_clicks=0),
                            style={"textAlign": "center", "marginTop": "10px"}
                        ),
                        dcc.Graph(
                            id="profile-view",
                            figure=make_empty_radar(),
                            style={"height": "450px", "marginTop": "20px"},
                        ),
                    ],
                ),



                html.Div(
                    style={"flex": "2"},
                    children=[
                        html.Div(
                            id="analytics-summary",
                            style={"marginBottom": "10px"},
                        ),
                        dcc.Graph(
                            id="score-bar",
                            figure=empty_bar_figure(),
                            style={"height": "300px", "marginTop": "10px"},
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# ---------- CALLBACK ----------
@app.callback(
    Output("profile-view", "figure"),
    Output("analytics-summary", "children"),
    Output("score-bar", "figure"),
    Output("pca-scatter", "figure"),
    Input("pca-scatter", "selectedData"),
    Input("pca-scatter", "clickData"),
    Input("profile-view", "clickData"),
    Input("clear-selection", "n_clicks"),
    State("pca-scatter", "figure"),
    State("profile-view", "figure"), 
)
def update_profile(selectedData, pca_clickData, radar_clickData, clear_clicks, current_pca_fig, current_radar_fig):
    ctx = dash.callback_context

    if not ctx.triggered:
        return make_empty_radar(), html.P("Select universities to see the summary."), empty_bar_figure(), dash.no_update

    prop_id = ctx.triggered[0]["prop_id"]          
    trigger_component, trigger_prop = prop_id.split(".")


    if trigger_component == "clear-selection":
        return make_empty_radar(), html.P("Select universities to see the summary."), empty_bar_figure(), pca_fig

    indices = []
    clicked_uni_id = None
    
    # PCA selection/click
    if trigger_component == "pca-scatter":
        # box/lasso selection
        if trigger_prop == "selectedData" and selectedData and "points" in selectedData:
            indices = [
                p["customdata"][0]
                for p in selectedData["points"]
                if p.get("customdata") is not None
            ]

        # single click
        elif trigger_prop == "clickData" and pca_clickData and "points" in pca_clickData and len(pca_clickData["points"]) > 0:
            pt = pca_clickData["points"][0]
            if pt.get("customdata") is not None:
                indices = [pt["customdata"][0]]
            else:
                # no safe global row_id -> ignore
                indices = []

    # radar click - preserve the current selection from the radar chart
    elif trigger_component == "profile-view":
        if current_radar_fig and "data" in current_radar_fig:
            for trace in current_radar_fig["data"]:
                trace_name = trace.get("name", "")
                if trace_name in ["Selection mean", "Global mean"]:
                    continue
                if "customdata" in trace and trace["customdata"]:
                    rid = trace["customdata"][0]
                    if rid not in indices:
                        indices.append(rid)
        
        # handle the click to highlight on PCA
        if radar_clickData and "points" in radar_clickData:
            clicked_point = radar_clickData["points"][0]
            
            if "curveNumber" in clicked_point and current_radar_fig:
                curve_num = clicked_point["curveNumber"]
                
                # get trace name from the current radar figure
                if "data" in current_radar_fig and curve_num < len(current_radar_fig["data"]):
                    trace = current_radar_fig["data"][curve_num]
                    trace_name = trace.get("name", "")
                                        
                    if trace_name and trace_name not in ["Selection mean", "Global mean"]:
                        uni_row = df[df["Institution Name"] == trace_name]
                        if not uni_row.empty:
                            clicked_uni_id = uni_row.iloc[0]["row_id"]

    if not indices:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


    sel = df[df["row_id"].isin(indices)]

    NUMERIC_COLS = FEATURES_PCA + ["QS Overall Score"]
    sel[NUMERIC_COLS] = sel[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")

    # ------------------ RADAR CHART ------------------
    radar_fig = go.Figure()

    # 1) Individual university profiles 
    max_individuals = 10
    for _, row in sel.head(max_individuals).iterrows():
        rid = int(row["row_id"])
        radar_fig.add_trace(
            go.Scatterpolar(
                r=[row[f] for f in FEATURES_PCA],
                theta=FEATURES_PCA,
                mode="lines+markers",
                fill=None,
                name=row["Institution Name"],
                opacity=0.5,
                customdata=[rid] * len(FEATURES_PCA),
            )
        )

    # 2) Selection mean profile
    mean_profile = sel[FEATURES_PCA].mean()
    radar_fig.add_trace(
        go.Scatterpolar(
            r=[mean_profile[f] for f in FEATURES_PCA],
            theta=FEATURES_PCA,
            fill="toself",
            name="Selection mean",
            opacity=0.7,
        )
    )

    # 3) Global mean profile
    global_mean = GLOBAL_MEAN

    # --- analytics: distances and feature differences ---

    # distance of each selected uni from global mean
    sel = sel.copy()
    sel["profile_distance"] = sel.apply(
        lambda row: compute_profile_distance(row, global_mean), axis=1
    )
    # sort by distance for neighbor queries
    sel = sel.sort_values("profile_distance")

    # ------------------ PCA UPDATE FOR RADAR CLICKS ------------------
    updated_pca = dash.no_update
    if trigger_component == "profile-view" and clicked_uni_id is not None:
        highlight_row = df[df["row_id"] == clicked_uni_id].iloc[0]
        custom = [[int(clicked_uni_id)]]

        updated_pca = go.Figure(current_pca_fig)
        
        # Layer 1: Large black outline (bottom layer)
        updated_pca.add_trace(
            go.Scatter(
                x=[highlight_row["pca_1"]],
                y=[highlight_row["pca_2"]],
                mode="markers",
                marker=dict(
                    size=35,
                    symbol="circle",
                    color="black",
                ),
                showlegend=False,
                hoverinfo="skip",
                customdata=custom,
            )
        )
        
        # Layer 2: White circle for contrast
        updated_pca.add_trace(
            go.Scatter(
                x=[highlight_row["pca_1"]],
                y=[highlight_row["pca_2"]],
                mode="markers",
                marker=dict(
                    size=30,
                    symbol="circle",
                    color="white",
                ),
                showlegend=False,
                hoverinfo="skip",
                customdata=custom,
            )
        )
        
        # Layer 3: Orange/red inner circle
        updated_pca.add_trace(
            go.Scatter(
                x=[highlight_row["pca_1"]],
                y=[highlight_row["pca_2"]],
                mode="markers",
                marker=dict(
                    size=22,
                    symbol="circle",
                    color="#FF6347",  # Tomato red
                ),
                name=f"Selected: {highlight_row['Institution Name']}",
                showlegend=False,
                hoverinfo="skip",
                customdata=custom,
            )
        )
        

    # ------------------ SINGLE-UNIVERSITY DETAIL MODE ------------------
    if len(sel) == 1:
        uni = sel.iloc[0]

        # Radar: only this university vs global mean
        radar_fig = go.Figure()

        rid = int(uni["row_id"])
        radar_fig.add_trace(
            go.Scatterpolar(
                r=[uni[f] for f in FEATURES_PCA],
                theta=FEATURES_PCA,
                mode="lines+markers",
                fill="toself",
                name=uni["Institution Name"],
                opacity=0.7,
                customdata=[rid] * len(FEATURES_PCA), 
            )
        )

        radar_fig.add_trace(
            go.Scatterpolar(
                r=[global_mean[f] for f in FEATURES_PCA],
                theta=FEATURES_PCA,
                mode="lines+markers",
                fill="toself",
                name="Global mean",
                opacity=0.3,
            )
        )

        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title=f"Profile of {uni['Institution Name']}",
        )

        # nearest neighbors to this university in feature space (excluding itself)
        k = 5
        others = df[df["row_id"] != uni["row_id"]].copy()
        others["profile_distance"] = others.apply(
            lambda row: compute_distance_between_rows(row, uni), axis=1
        )
        neighbors = others.sort_values("profile_distance").head(k)

        # table for this university's features
        feature_rows = []
        for f in FEATURES_PCA:
            feature_rows.append(
                html.Tr(
                    [
                        html.Td(f),
                        html.Td(f"{uni[f]:.1f}" if pd.notna(uni[f]) else "N/A"),
                        html.Td(f"{global_mean[f]:.1f}" if pd.notna(global_mean[f]) else "N/A"),
                    ]
                )
            )

        uni_feature_table = html.Table(
            [
                html.Tr(
                    [
                        html.Th("Feature"),
                        html.Th("Selected university"),
                        html.Th("Global mean"),
                    ]
                )
            ]
            + feature_rows,
            style={"borderCollapse": "collapse", "width": "100%", "fontSize": "12px"},
        )

        # neighbors table
        nn_header = html.Tr(
            [
                html.Th("Neighbor institution"),
                html.Th("Location"),
                html.Th("QS Overall"),
                html.Th("2025 Rank"),
                html.Th("Profile distance"),
            ]
        )
        nn_rows = []
        for _, row in neighbors.iterrows():
            nn_rows.append(
                html.Tr(
                    [
                        html.Td(row["Institution Name"]),
                        html.Td(row["Location"]),
                        html.Td(
                            f"{row['QS Overall Score']:.1f}"
                            if pd.notna(row["QS Overall Score"])
                            else "N/A"
                        ),
                        html.Td(
                            row["2025 Rank"] if pd.notna(row["2025 Rank"]) else "N/A"
                        ),
                        html.Td(f"{row['profile_distance']:.2f}"),
                    ]
                )
            )

        nn_table = html.Table(
            [nn_header] + nn_rows,
            style={
                "borderCollapse": "collapse",
                "width": "100%",
                "fontSize": "12px",
            },
        )

        info_panel = html.Div(
            children=[
                html.H3(f"Detail view: {uni['Institution Name']}"),
                html.Ul(
                    [
                        html.Li(f"Location: {uni['Location']}"),
                        html.Li(
                            f"QS Overall Score: {uni['QS Overall Score']:.1f}"
                            if pd.notna(uni["QS Overall Score"])
                            else "QS Overall Score: N/A"
                        ),
                        html.Li(
                            f"2025 Rank: {uni['2025 Rank']}"
                            if pd.notna(uni["2025 Rank"])
                            else "2025 Rank: N/A"
                        ),
                        html.Li(
                            f"Profile distance from global mean: {uni['profile_distance']:.2f}"
                        ),
                    ],
                    style={"paddingLeft": "20px"},
                ),
                html.H4("Feature comparison (university vs global mean)"),
                uni_feature_table,
                html.H4(f"Top {k} similar universities (by profile distance)"),
                nn_table,
            ],
            style={
                "backgroundColor": "#fafafa",
                "padding": "8px 10px",
                "border": "1px solid #eee",
                "borderRadius": "4px",
            },
        )

        # simple bar in detail mode: show this uni's features
        bar_fig = go.Figure()
        bar_fig.add_trace(
            go.Bar(
                y=FEATURES_PCA,
                x=[uni[f] for f in FEATURES_PCA],
                orientation="h",
            )
        )
        bar_fig.update_layout(
            title="Feature scores of selected university",
            xaxis_title="Score",
            yaxis_title="Feature",
            showlegend=False,
            margin=dict(l=120),
        )

        return radar_fig, info_panel, bar_fig, updated_pca
    
    else:
        # ------------------ MULTIPLE UNIVERSITIES MODE ------------------
        # features that most distinguish selection vs global mean
        feature_diffs = compute_feature_differences(mean_profile, global_mean)
        top_features = feature_diffs[:3]  # top 3 distinguishing features

        radar_fig.add_trace(
            go.Scatterpolar(
                r=[global_mean[f] for f in FEATURES_PCA],
                theta=FEATURES_PCA,
                fill="toself",
                name="Global mean",
                opacity=0.3,
            )
        )

        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title=f"Profiles of Selected Universities (n={len(sel)})",
        )

        # ------------------ SELECTION INFO ------------------

        avg_qs = sel["QS Overall Score"].mean()
        avg_acad = sel["Academic Reputation"].mean()
        avg_dist = sel["profile_distance"].mean()

        table_header = html.Tr(
            [
                html.Th("Institution"),
                html.Th("Location"),
                html.Th("QS Overall"),
                html.Th("2025 Rank"),
                html.Th("Profile distance"),
            ]
        )

        table_rows = []
        for _, row in sel.iterrows():
            table_rows.append(
                html.Tr(
                    [
                        html.Td(row["Institution Name"]),
                        html.Td(row["Location"]),
                        html.Td(
                            f"{row['QS Overall Score']:.1f}"
                            if pd.notna(row["QS Overall Score"])
                            else "N/A"
                        ),
                        html.Td(
                            row["2025 Rank"] if pd.notna(row["2025 Rank"]) else "N/A"
                        ),
                        html.Td(f"{row['profile_distance']:.2f}"),
                    ]
                )
            )

        selection_table = html.Table(
            [table_header] + table_rows,
            style={
                "borderCollapse": "collapse",
                "width": "100%",
                "fontSize": "12px",
            },
        )
        info_panel = html.Div(
            children=[
                html.H3("Selected Universities Overview"),
                html.Ul(
                    [
                        html.Li(f"Number of universities: {len(sel)}"),
                        html.Li(
                            f"Average QS Overall Score: {avg_qs:.2f}"
                            if pd.notna(avg_qs)
                            else "Average QS Overall Score: N/A"
                        ),
                        html.Li(
                            f"Average Academic Reputation: {avg_acad:.2f}"
                            if pd.notna(avg_acad)
                            else "Average Academic Reputation: N/A"
                        ),
                        html.Li(
                            "Most distinctive features vs global mean: "
                            + ", ".join(
                                f"{fd['feature']} ({fd['diff']:+.1f})"
                                for fd in top_features
                            )
                        ),
                        html.Li(
                            f"Average profile distance from global mean: {avg_dist:.2f}"
                        ),
                    ],
                    style={"paddingLeft": "20px"},
                ),
                html.H4("Selected universities:"),
                html.Div(
                    selection_table,
                    style={
                        "maxHeight": "230px",
                        "overflowY": "auto",
                        "border": "1px solid #ddd",
                        "padding": "4px",
                    },
                ),
            ],
            style={
                "backgroundColor": "#fafafa",
                "padding": "8px 10px",
                "border": "1px solid #eee",
                "borderRadius": "4px",
            },
        )

        diffs_for_bar = feature_diffs[:5]

        bar_fig = go.Figure()

        bar_fig.add_trace(
            go.Bar(
                y=[d["feature"] for d in diffs_for_bar],
                x=[d["diff"] for d in diffs_for_bar],
                orientation="h",
                marker_color=[
                    "#2ca02c" if d["diff"] >= 0 else "#d62728" for d in diffs_for_bar
                ],
            )
        )

        bar_fig.update_layout(
            title="Top 5 feature differences (selection − global)",
            xaxis_title="Difference",
            yaxis_title="Feature",
            showlegend=False,
            margin=dict(l=120),
        )

        return radar_fig, info_panel, bar_fig, updated_pca

# ---------- MAIN ----------


if __name__ == "__main__":
    app.run(debug=True)