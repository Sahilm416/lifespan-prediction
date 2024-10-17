import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
import plotly.express as px
import plotly.graph_objects as go

# Load data
@st.cache_data
def load_data():
    try:
        tx_data = pd.read_csv("OnlineRetail.csv", encoding="cp1252")
        tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'], format="%m/%d/%Y %H:%M").dt.date
        tx_data = tx_data[pd.notnull(tx_data['CustomerID'])]
        tx_data = tx_data[(tx_data['Quantity'] > 0)]
        tx_data['Total_Sales'] = tx_data['Quantity'] * tx_data['UnitPrice']
        return tx_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.title('Customer Lifetime Value Calculator')

    tx_data = load_data()
    if tx_data is None:
        return

    # Summary data
    lf_tx_data = summary_data_from_transaction_data(tx_data, 'CustomerID', 'InvoiceDate', monetary_value_col='Total_Sales', observation_period_end='2011-12-9')

    # Calculate CLV early
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(lf_tx_data['frequency'], lf_tx_data['recency'], lf_tx_data['T'])

    ggf = GammaGammaFitter(penalizer_coef=0)
    ggf.fit(lf_tx_data[lf_tx_data['frequency'] > 0]['frequency'], 
            lf_tx_data[lf_tx_data['frequency'] > 0]['monetary_value'])

    time_period = 12  # Default to 12 months
    discount_rate = 0.01  # Default discount rate

    lf_tx_data['CLV'] = ggf.customer_lifetime_value(
        bgf, 
        lf_tx_data['frequency'],
        lf_tx_data['recency'],
        lf_tx_data['T'],
        lf_tx_data['monetary_value'],
        time=time_period,
        discount_rate=discount_rate
    )

    # Display dataset summary
    st.subheader('Dataset Summary')
    st.write(f"Number of unique customers: {lf_tx_data.index.nunique()}")

    # Histogram of purchase frequencies
    st.subheader('Purchase Frequency Histogram')
    fig, ax = plt.subplots()
    ax.hist(lf_tx_data['frequency'], bins=50)
    st.pyplot(fig)

    # Percentage of one-time buyers
    one_time_buyers = round(sum(lf_tx_data['frequency'] == 0) / len(lf_tx_data) * 100, 2)
    st.write(f"Percentage of customers who purchased only once: {one_time_buyers}%")

    # Frequency/Recency Analysis Using the BG/NBD Model
    st.subheader('Frequency/Recency Analysis Using the BG/NBD Model')
    st.write(bgf.summary)

    # 1. CLV Distribution
    st.subheader('CLV Distribution')
    fig_clv_dist = px.histogram(lf_tx_data, x='CLV', nbins=50, title='Distribution of Customer Lifetime Value')
    st.plotly_chart(fig_clv_dist)

    # 2. Scatter plot of CLV vs Frequency
    st.subheader('CLV vs Purchase Frequency')
    fig_clv_freq = px.scatter(lf_tx_data, x='frequency', y='CLV', 
                              hover_data=['monetary_value', 'recency'],
                              title='CLV vs Purchase Frequency')
    st.plotly_chart(fig_clv_freq)

    # 3. Top 10 Customers by CLV
    st.subheader('Top 10 Customers by CLV')
    top_10_customers = lf_tx_data.nlargest(10, 'CLV')
    fig_top_10 = go.Figure(data=[
        go.Bar(name='CLV', x=top_10_customers.index, y=top_10_customers['CLV']),
        go.Bar(name='Frequency', x=top_10_customers.index, y=top_10_customers['frequency'])
    ])
    fig_top_10.update_layout(barmode='group', title='Top 10 Customers: CLV and Purchase Frequency')
    st.plotly_chart(fig_top_10)

    # 4. Interactive Customer Segmentation
    st.subheader('Customer Segmentation')
    x_axis = st.selectbox("Select X-axis", options=['frequency', 'recency', 'T', 'monetary_value', 'CLV'])
    y_axis = st.selectbox("Select Y-axis", options=['CLV', 'frequency', 'recency', 'T', 'monetary_value'])
    
    fig_segmentation = px.scatter(lf_tx_data, x=x_axis, y=y_axis, 
                                  color='CLV', hover_data=['frequency', 'recency', 'monetary_value'],
                                  title=f'Customer Segmentation: {x_axis.capitalize()} vs {y_axis.capitalize()}')
    st.plotly_chart(fig_segmentation)


    # Predict future transactions
    st.subheader('Predict Future Transactions')
    t = st.slider("Select number of days for prediction", min_value=1, max_value=30, value=10)
    lf_tx_data['pred_num_txn'] = round(
        bgf.conditional_expected_number_of_purchases_up_to_time(t, lf_tx_data['frequency'], lf_tx_data['recency'],
                                                                lf_tx_data['T']), 2)
    st.write(lf_tx_data.sort_values(by='pred_num_txn', ascending=False).head(10))

    # Customer's future transaction prediction
    st.subheader("Customer's Future Transaction Prediction")
    customer_id = st.selectbox("Select a customer ID", options=lf_tx_data.index.tolist())
    individual = lf_tx_data.loc[customer_id]
    predicted_transactions = bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])
    st.write(f"Predicted transactions for customer {customer_id} in the next {t} days: {predicted_transactions:.2f}")

    # Train Gamma-Gamma model
    st.subheader('Train Gamma-Gamma Model')
    shortlisted_customers = lf_tx_data[lf_tx_data['frequency'] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0)
    ggf.fit(shortlisted_customers['frequency'], shortlisted_customers['monetary_value'])
    st.write(ggf.summary)

    # Calculate Customer Lifetime Value
    st.subheader('Calculate Customer Lifetime Value')
    time_period = st.slider("Select time period for CLV calculation (months)", min_value=1, max_value=24, value=12)
    discount_rate = st.slider("Select discount rate for CLV calculation", min_value=0.01, max_value=0.20, value=0.01, step=0.01)
    
    lf_tx_data['CLV'] = round(
        ggf.customer_lifetime_value(bgf, lf_tx_data['frequency'], lf_tx_data['recency'], lf_tx_data['T'],
                                    lf_tx_data['monetary_value'], time=time_period, discount_rate=discount_rate), 2)
    st.write(lf_tx_data.sort_values(by='CLV', ascending=False).head(10))

    # Individual customer CLV prediction
    st.subheader("Individual Customer CLV Prediction")
    selected_customer_id = st.selectbox("Select a customer ID for CLV prediction", options=lf_tx_data.index.tolist())
    selected_customer = lf_tx_data.loc[selected_customer_id]
    
    # Create a single-row DataFrame for the selected customer
    customer_data = pd.DataFrame({
        'frequency': [selected_customer['frequency']],
        'recency': [selected_customer['recency']],
        'T': [selected_customer['T']],
        'monetary_value': [selected_customer['monetary_value']]
    })
    
    individual_clv = ggf.customer_lifetime_value(
        bgf, 
        customer_data['frequency'],
        customer_data['recency'],
        customer_data['T'],
        customer_data['monetary_value'],
        time=time_period,
        discount_rate=discount_rate
    )
    
    st.write(f"Predicted CLV for customer {selected_customer_id} over {time_period} months: ${individual_clv.iloc[0]:.2f}")

if __name__ == "__main__":
    main()