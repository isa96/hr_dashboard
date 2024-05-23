import gradio as gr

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

import pickle
from lime.lime_tabular import LimeTabularExplainer

# pull and preprocess data
link_to_data = "https://raw.githubusercontent.com/hadimaster65555/dataset_for_teaching/main/dataset/hr_analytics_turnover_dataset/HR_comma_sep.csv"
raw_data = pd.read_csv(link_to_data)
raw_data.reset_index(inplace=True)
raw_data = raw_data.rename(columns={"index": "employee_id", "sales": "roles"})


def preprocess_data_for_lime(data):
    X_data = data.drop(['employee_id', 'left'], axis=1).copy()
    y_data = data['left'].copy()
    X_data['salary'] = X_data['salary'].replace({"low": 1, "medium": 2, "high": 3})
    X_data = pd.get_dummies(X_data, dtype=float).drop(columns="roles_IT")
    X_train, _, _, _ = train_test_split(
        X_data,
        y_data,
        test_size=0.3,
        random_state=65555,
        stratify=y_data
    )
    return X_train


def explainer_generator(train_data):
    explainer = LimeTabularExplainer(
        train_data,
        feature_names=train_data.columns,
        class_names=['Stay', 'Left'],
        discretize_continuous=False,
        random_state=65555
    )
    return explainer


def explainer_to_lime_importance(explainer, employee_data, model):
    lime_res = explainer.explain_instance(employee_data.iloc[0], model.predict_proba, num_features=len(employee_data.columns))
    importance_res = pd.DataFrame(lime_res.as_list()).rename(columns={0: "variable", 1: "value"})
    fig = px.bar(
        importance_res,
        y="variable",
        x="value",
        color="variable", orientation="h")
    return fig


def raw_pred_to_class_pred(data, model):
    employee_att_stat = ["Stay", "Left"]
    pred_result = model.predict_proba(data)[0]
    return dict(zip(employee_att_stat, pred_result))


automl = pickle.load(open('./model/automl.pkl', 'rb'))

explainer_data = preprocess_data_for_lime(raw_data)
explainer_obj = explainer_generator(explainer_data)


# search user function
def search_data(user_id):
    alt_data = raw_data.copy()
    alt_data['salary'] = alt_data['salary'].replace({"low": 1, "medium": 2, "high": 3})
    alt_data = pd.get_dummies(alt_data, dtype=float).drop(columns="roles_IT")

    iseng_res = automl.predict(alt_data)

    employee_data = alt_data[alt_data.employee_id == int(user_id)].copy()
    importance_fig = explainer_to_lime_importance(
        explainer_obj,
        employee_data.drop(columns=["employee_id", "left"]),
        automl
    )
    alt_data['pred'] = iseng_res

    pred_result = raw_pred_to_class_pred(data=employee_data.drop(columns=["employee_id", "left"]), model=automl)

    employee_data = raw_data[raw_data.employee_id == int(user_id)].copy()

    employee_data["left"] = employee_data["left"].replace({0: "Stay", 1: "Left"})
    employee_data["promotion_last_5years"] = employee_data["promotion_last_5years"].replace({0: "Not Promoted", 1: "Promoted"})
    employee_data["Work_accident"] = employee_data["Work_accident"].replace({0: "No", 1: "Yes"})

    promotion_data = str(employee_data.promotion_last_5years.to_list()[0])
    satisfaction_level_data = str(employee_data.satisfaction_level.to_list()[0])
    last_evaluation_data = str(employee_data.last_evaluation.to_list()[0])
    role_data = str(employee_data.roles.to_list()[0])
    salary_data = str(employee_data.salary.to_list()[0])
    number_project_data = str(employee_data.number_project.to_list()[0])
    average_hours_data = str(employee_data.average_montly_hours.to_list()[0])
    time_spend_company_data = str(employee_data.time_spend_company.to_list()[0])
    work_accident_data = str(employee_data.Work_accident.to_list()[0])

    return {
        user_data: f"## User ID: {user_id}",
        attrition_pred: pred_result,
        imp_fig: importance_fig,
        employee_role: role_data,
        employee_salary: salary_data,
        employee_status: promotion_data,
        satisfaction_level: f"{satisfaction_level_data} out of 1",
        last_evaluation: f"{last_evaluation_data} out of 1",
        number_project: f"{number_project_data} Projects",
        avg_hours: f"{average_hours_data} Hours",
        time_spend: f"{time_spend_company_data} Years",
        employee_work_accident: work_accident_data
    }


# single value for median satisfaction level
def indicator_satisfaction_level():
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=raw_data['satisfaction_level'].median(),
        domain={'row': 0, 'column': 1}))

    fig.update_layout(
        template={
            'data': {'indicator': [{
                'title': {'text': "Median Satisfaction Level"},
                'mode': "number"}]
            }
        }
    )
    return fig


# pie chart for visualize roles freq data who not left company
def bar_chart_roles():
    fig = px.bar(
        raw_data[raw_data.left != 1].groupby('roles', as_index=False).size().sort_values(by="size", ascending=False),
        y="size", x="roles", color="roles")
    return fig


# pie chart for visualize salary freq data who not left company
def pie_chart_salary():
    fig = px.pie(raw_data[raw_data.left != 1].groupby('salary', as_index=False).size(), values="size", names="salary")
    return fig


# pie chart for visualize promotion freq data who not left company
def pie_chart_promotion():
    fig = px.pie(
        raw_data[raw_data.left != 1].groupby('promotion_last_5years', as_index=False).size(),
        values="size",
        names="promotion_last_5years"
    )
    return fig


def pie_chart_left():
    fig = px.pie(
        raw_data.groupby('left', as_index=False).size().replace({0: "Stay", 1: "Left"}),
        values="size", names="left"
    )
    return fig


with gr.Blocks() as demo:
    gr.Markdown("""
    # HR Dashboard
    """)
    with gr.Tab("Main Dashboard"):
        gr.Markdown("# Main Dashboard")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Median Satisfaction Level")
                gr.Plot(indicator_satisfaction_level)
            with gr.Column():
                gr.Markdown("### Total Employees by Employment Status")
                gr.Plot(pie_chart_left)
        gr.Markdown("## Existing Employees Statistics")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Total Employees by Roles")
                gr.Plot(bar_chart_roles)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Total Employees by Salary")
                gr.Plot(pie_chart_salary)
            with gr.Column():
                gr.Markdown("### Total Promoted Employees in The Last 5 Years")
                gr.Plot(pie_chart_promotion)
    with gr.Tab("Employee Inspector"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# Data Input")
                id_input = gr.Textbox(
                    placeholder="Input Employee ID from 1 to 14999",
                    label="Employee ID",
                    show_label=True,
                )
                search_button = gr.Button("Search")

            with gr.Column(scale=3):
                gr.Markdown("# Employee Description")
                user_data = gr.Markdown("## Please input employee ID")
                with gr.Row():
                    attrition_pred = gr.Label("No Data", label="Predicted Attrition Status")
                with gr.Row():
                    imp_fig = gr.Plot()
                with gr.Row():
                    employee_role = gr.Label("No Data", label="User ID and Role")
                    employee_salary = gr.Label("No Data", label="Salary Level")
                    employee_work_accident = gr.Label("No Data", label="Work Accident")
                with gr.Row():
                    with gr.Column():
                        employee_status = gr.Label("No Data", label="Promotion Status")
                    with gr.Column():
                        satisfaction_level = gr.Label("No Data", label="Satisfaction Level")
                    with gr.Column():
                        last_evaluation = gr.Label("No Data", label="Last Evaluation")
                with gr.Row():
                    with gr.Column():
                        number_project = gr.Label("No Data", label="Number of Projects")
                    with gr.Column():
                        avg_hours = gr.Label("No Data", label="Average Monthly Hours")
                    with gr.Column():
                        time_spend = gr.Label("No Data", label="Time Spend in Company")

    search_button.click(fn=search_data, inputs=id_input, outputs=[
        user_data, attrition_pred, imp_fig,
        employee_role, employee_salary, employee_status,
        satisfaction_level, last_evaluation, number_project, avg_hours,
        time_spend, employee_work_accident
    ])


if __name__ == "__main__":
    demo.launch()
