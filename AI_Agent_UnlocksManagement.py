import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Streamlit App Title
st.title("Interactive Token Unlock Schedule with Market Depth Provisioning")

# User Inputs
st.sidebar.header("Tokenomics Inputs")

# General Parameters
max_supply = st.sidebar.number_input("Maximum Supply (tokens)", value=1_000_000_000, step=100_000_000)
initial_token_price = st.sidebar.number_input("Initial Token Price (USD)", value=0.1, step=0.01)
token_price = st.sidebar.number_input("Token Price (USD, Price Model)", value=0.1, step=0.01)

# Offset Simulation
offset_month = st.sidebar.number_input("Offset Simulation Start Month", value=0, min_value=0, step=1)

# Price Model Selection
st.sidebar.header("Price Model")
price_model = st.sidebar.radio("Choose Price Model", ("Constant Price", "Stochastic Price (Black-Scholes)"))

if price_model == "Stochastic Price (Black-Scholes)":
    mu = st.sidebar.number_input("Expected Return (mu)", value=0.05, step=0.01)
    sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01)
    time_horizon = 40  # 40 months
    dt = 1 / 12
    np.random.seed(42)

    # Generate 100 stochastic simulations starting from offset_month
    simulations = []
    for _ in range(100):
        stochastic_prices = [token_price] * max(offset_month, 1)  # Ensure at least one value
        for t in range(max(offset_month, 1), time_horizon):
            random_shock = np.random.normal(0, 1)
            price = stochastic_prices[-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_shock)
            stochastic_prices.append(price)
        simulations.append(stochastic_prices)

    # Convert simulations to numpy array for processing
    simulations = np.array(simulations)

    # Ensure the paths remain in proper order (lower, median, upper) at each time step
    sorted_simulations = np.sort(simulations, axis=0)

    # Select three scenarios: lower, median, and upper paths
    lower_path = sorted_simulations[0]
    median_path = sorted_simulations[len(sorted_simulations) // 2]
    upper_path = sorted_simulations[-1]

else:
    time_horizon = 40  # Fixed duration
    lower_path = median_path = upper_path = [token_price] * time_horizon

# Ensure paths start from offset_month and align with total horizon
lower_path = np.pad(lower_path[offset_month:], (offset_month, 0), mode='constant', constant_values=token_price)[:40]
median_path = np.pad(median_path[offset_month:], (offset_month, 0), mode='constant', constant_values=token_price)[:40]
upper_path = np.pad(upper_path[offset_month:], (offset_month, 0), mode='constant', constant_values=token_price)[:40]

# Bear Market Periods
st.sidebar.header("Bear Market Periods")
bear_market_periods = st.sidebar.text_input("Bear Market Periods (e.g., [(10, 16), (28, 34)])", value="[(10, 16), (28, 34)]")
bear_market_coefficient = st.sidebar.number_input("Bear Market Sell Pressure Coefficient", value=1.5, step=0.1)
try:
    bear_market_periods = eval(bear_market_periods)
except:
    st.sidebar.error("Invalid format for bear market periods. Use [(start, end), ...]")

# Vesting Schedule Parameters
st.sidebar.header("Vesting Schedule Parameters")
vesting_columns = ["Category", "TGE (%)", "Unlock (%)", "Lock-up (months)", "Start Month", "End Month", "Color", "Default SP (%)", "Triggered SP (%)", "Trigger ROI (%)"]
vesting_data = [
    ["Pre-Seed", 0.0, 0.005, 0, 1, 30, "#0000FF", 50, 95, 110],
    ["Seed", 0.0, 0.004, 0, 1, 24, "#008000", 50, 95, 110],
    ["Public Sale", 0.01, 0.0, 0, 0, 0, "#FFA500", 50, 95, 110],
    ["Team/Founders", 0.0, 0.003, 12, 13, 40, "#800080", 50, 95, 110],
    ["Treasury", 0.0, 0.002, 0, 1, 35, "#00FFFF", 50, 95, 110],
    ["Airdrop", 0.015, 0.0, 0, 0, 0, "#FF0000", 50, 95, 110],
    ["Marketing", 0.03, 0.005, 3, 4, 9, "#FFC0CB", 50, 95, 110],
    ["Liquidity", 0.01, 0.0, 0, 0, 0, "#808080", 50, 95, 110],
]
vesting_df = pd.DataFrame(vesting_data, columns=vesting_columns)

# Editable Vesting Schedule
st.write("### Edit Vesting Schedule")
edited_vesting_data = []
for index, row in vesting_df.iterrows():
    cols = st.columns(len(vesting_columns))
    edited_row = []
    for i, col in enumerate(cols):
        unique_key = f"{vesting_columns[i]}_{index}"
        if vesting_columns[i] == "Color":
            value = col.color_picker(f"{vesting_columns[i]} ({index})", value=row[i], key=unique_key)
        else:
            value = col.text_input(f"{vesting_columns[i]} ({index})", value=row[i], key=unique_key)
            try:
                value = float(value) if i > 0 else value
            except ValueError:
                pass
        edited_row.append(value)
    edited_vesting_data.append(edited_row)
vesting_df = pd.DataFrame(edited_vesting_data, columns=vesting_columns)

# Dynamic Market Depth
st.sidebar.header("Dynamic Market Depth")
market_depth_threshold = st.sidebar.number_input("Market Depth Threshold (USD)", value=1_000_000, step=100_000)

# Liquidity Provisioning
st.sidebar.header("Liquidity Provisioning")
liquidity_provisioning = st.sidebar.text_input(
    "Liquidity Provisioning Additions (e.g., {15: 500000, 25: 750000})",
    value="{15: 500000, 25: 750000}"
)
try:
    liquidity_provisioning = eval(liquidity_provisioning)
except:
    st.sidebar.error("Invalid format for liquidity provisioning. Use {month: amount, ...}")

dynamic_market_depth = [market_depth_threshold]
for i in range(1, 40):
    added_liquidity = liquidity_provisioning.get(i, 0)
    dynamic_market_depth.append(dynamic_market_depth[-1] + added_liquidity)

# Rewards Allocation
st.sidebar.header("Rewards Allocation")
reward_allocation_percentage = st.sidebar.slider("Rewards Allocation (% of Total Supply)", 0.0, 100.0, 5.0, 0.1)
logistic_center = st.sidebar.slider("Logistic Center (Months)", 0, 40, 20, 1)
logistic_steepness = st.sidebar.slider("Logistic Steepness", 0.1, 10.0, 1.0, 0.1)


# Tokens Non Vendus Visualization Toggle
show_unsold_tokens = st.sidebar.checkbox("Show Unsold Tokens", value=False)

# Unsold Tokens Logic
unsold_tokens = np.zeros(40)  # Adjusted to consider only months before offset

# Collect Unsold Tokens Before Offset
for _, entry in vesting_df.iterrows():
    if entry["Unlock (%)"] > 0:
        for month in range(0, offset_month):
            price_roi = (median_path[month] / initial_token_price) * 100
            sell_pressure = entry["Default SP (%)"] / 100
            if price_roi > entry["Trigger ROI (%)"]:
                sell_pressure = entry["Triggered SP (%)"] / 100
            unsold_tokens[month] += entry["Unlock (%)"] * (1 - sell_pressure)

# Unsold Tokens Redistribution Mode
st.sidebar.header("Unsold Tokens Management")
tokens_redistribution_method = st.sidebar.radio("Unsold Tokens Redistribution:", ["Lissage (12 mois)", "Dépôt Discret"])
if tokens_redistribution_method == "Dépôt Discret":
    deposit_month = st.sidebar.slider("Deposit Month", offset_month, 39, offset_month + 1)
    deposit_spread = st.sidebar.slider("Deposit Spread (Months)", 1, 3, 1)

# Redistribute Unsold Tokens with Re-Roll Logic
unsold_tokens_redistributed = np.zeros(40)
for month in range(offset_month):
    if unsold_tokens[month] > 0:
        # Distribute unsold tokens to the following months (Lissage or Dépôt Discret)
        if tokens_redistribution_method == "Lissage (12 mois)":
            for future_month in range(offset_month, min(offset_month + 12, 40)):
                redistributed_tokens = unsold_tokens[month] / 12

                # Apply sale rules to redistributed tokens
                price_roi = (median_path[future_month - offset_month] / initial_token_price) * 100
                sell_pressure = 0.5  # Default fallback if no entry
                for _, entry in vesting_df.iterrows():  # Iterate categories to apply sale rules
                    sell_pressure = entry["Default SP (%)"] / 100
                    if price_roi > entry["Trigger ROI (%)"]:
                        sell_pressure = entry["Triggered SP (%)"] / 100

                # Calculate sold tokens and add them to the schedule
                sold_tokens = redistributed_tokens * sell_pressure
                unsold_tokens_redistributed[future_month] += sold_tokens

        elif tokens_redistribution_method == "Dépôt Discret":
            total_unsold = unsold_tokens[month]
            for i in range(deposit_spread):
                target_month = deposit_month + i
                if target_month < 40:
                    redistributed_tokens = total_unsold / deposit_spread

                    # Apply sale rules to redistributed tokens
                    price_roi = (median_path[target_month - offset_month] / initial_token_price) * 100
                    sell_pressure = 0.5  # Default fallback if no entry
                    for _, entry in vesting_df.iterrows():  # Iterate categories to apply sale rules
                        sell_pressure = entry["Default SP (%)"] / 100
                        if price_roi > entry["Trigger ROI (%)"]:
                            sell_pressure = entry["Triggered SP (%)"] / 100

                    # Calculate sold tokens and add them to the schedule
                    sold_tokens = redistributed_tokens * sell_pressure
                    unsold_tokens_redistributed[target_month] += sold_tokens





# Apply Redistribution to Unlocks
allocations = {}
for _, entry in vesting_df.iterrows():
    schedule = [0] * 40  # Initialize 40 months
    if entry["TGE (%)"] > 0:
        schedule[0] = entry["TGE (%)"]
    if entry["Unlock (%)"] > 0:
        for month in range(max(offset_month, int(entry["Start Month"])), min(40, int(entry["End Month"]) + 1)):
            price_roi = (median_path[month - offset_month] / initial_token_price) * 100
            sell_pressure = entry["Default SP (%)"] / 100
            if price_roi > entry["Trigger ROI (%)"]:
                sell_pressure = entry["Triggered SP (%)"] / 100
            schedule[month] += entry["Unlock (%)"] * sell_pressure
    schedule = [val if idx >= offset_month else 0 for idx, val in enumerate(schedule)]
    allocations[entry["Category"]] = {"color": entry["Color"], "unlock_schedule": schedule}


# Add Checkbox to Exclude Unlocks from Month 0
exclude_month_0 = st.sidebar.checkbox("Exclude Unlocks from Month 0", value=False)
if exclude_month_0:
    for name, data in allocations.items():
        data["unlock_schedule"][0] = 0  # Set month 0 unlocks to 0

# Add Unsold Tokens Redistribution
allocations["Unsold Tokens"] = {
    "color": "gray",
    "unlock_schedule": unsold_tokens_redistributed.tolist()
}

# Total unlocks in tokens
total_unlocks_tokens = np.zeros(40)
for data in allocations.values():
    total_unlocks_tokens += np.array(data["unlock_schedule"]) * max_supply  # Include all categories including Rewards

# Total unlocks in USD
lower_unlocks_usd = total_unlocks_tokens * lower_path
median_unlocks_usd = total_unlocks_tokens * median_path
upper_unlocks_usd = total_unlocks_tokens * upper_path

dynamic_market_depth = np.array(dynamic_market_depth)

# Add Rewards Allocation
x = np.arange(40)
logistic_curve = 1 / (1 + np.exp(-logistic_steepness * (x - logistic_center)))
logistic_curve = logistic_curve / logistic_curve.sum() * (reward_allocation_percentage / 100)

allocations["Rewards"] = {
    "color": "#FFD700",  # Gold color for rewards
    "unlock_schedule": logistic_curve.tolist()
}

# Plot Unlock Schedule
fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(40)
for name, data in allocations.items():
    unlock_usd = np.array(data["unlock_schedule"]) * max_supply * median_path
    ax.bar(range(40), unlock_usd, bottom=bottom, color=data["color"], label=name, alpha=0.7)
    bottom += unlock_usd

# Unsold Tokens Visualization
if show_unsold_tokens:
    unsold_usd = unsold_tokens_redistributed[:40] * median_path
    ax.bar(range(40), unsold_usd, bottom=bottom, color="gray", label="Unsold Tokens", alpha=0.5)

# Overflow Hatching
excess = bottom - dynamic_market_depth
ax.bar(
    range(40),
    excess.clip(min=0),
    bottom=dynamic_market_depth,
    color='none',
    edgecolor='red',
    hatch='//',
    label="Overflow"
)

# Market Depth
ax.step(range(40), dynamic_market_depth, where="mid", color="red", linestyle="--", label="Market Depth")

# Enveloppe des prix (Lower, Median, Upper)
ax2 = ax.twinx()
ax2.fill_between(range(40), lower_path, upper_path, color="blue", alpha=0.2, label="Price Envelope")
ax2.plot(range(40), lower_path, color="green", linestyle="--", linewidth=2, label="Lower Path")
ax2.plot(range(40), median_path, color="blue", linestyle="-", linewidth=3, label="Median Path")
ax2.plot(range(40), upper_path, color="orange", linestyle="--", linewidth=2, label="Upper Path")

ax2.set_ylabel("Token Price (USD)", color="blue")
ax2.tick_params(axis='y', labelcolor='blue')

# Finalize Plot
ax.set_title("Token Unlock Schedule with Unsold Tokens, Price Envelope, and Market Depth")
ax.set_xlabel("Months")
ax.set_ylabel("Unlock Value (USD)")
ax.legend(loc="upper left")
ax.grid(False)
ax2.legend(loc="upper right")

st.pyplot(fig)

# Range slider below the graph for cumulative overflow
st.write("### Select Range for Cumulative Overflow")
range_start, range_end = st.slider("Select Month Range:", 0, 39, (0, 39))

st.write("### Price Path Selection for Overflow")
selected_path = st.radio("Choose Price Path", ("Median", "Lower", "Upper"))

# Calculate Overflow based on selected price path
if selected_path == "Median":
    selected_unlocks_usd = median_unlocks_usd
    roi = [(price / initial_token_price - 1) * 100 for price in median_path]
elif selected_path == "Lower":
    selected_unlocks_usd = lower_unlocks_usd
    roi = [(price / initial_token_price - 1) * 100 for price in lower_path]
else:
    selected_unlocks_usd = upper_unlocks_usd
    roi = [(price / initial_token_price - 1) * 100 for price in upper_path]

overflow = [max(0, selected_unlocks_usd[i] - dynamic_market_depth[i]) for i in range(40)]

# Include Rewards in Cumulative Overflow
rewards_usd = np.array(allocations["Rewards"]["unlock_schedule"]) * max_supply * median_path
overflow_with_rewards = [max(0, (selected_unlocks_usd[i] + rewards_usd[i]) - dynamic_market_depth[i]) for i in range(40)]

cumulative_overflow = sum(overflow_with_rewards[range_start:range_end + 1])
st.write(f"**Cumulative Overflow (USD):** {cumulative_overflow:,.2f}")

# Plot ROI and Bear Market Periods
fig2, ax3 = plt.subplots(figsize=(12, 6))

# Primary axis: Overflow
ax3.fill_between(range(40), 0, overflow_with_rewards, color='red', alpha=0.7, linewidth=2)
ax3.set_ylabel("Overflow (USD)", color='red')
ax3.tick_params(axis='y', labelcolor='red')
ax3.set_ylim(bottom=-0.05, top=max(overflow_with_rewards) * 1.2)

# Secondary axis: ROI
ax4 = ax3.twinx()
ax4.plot(range(40), roi, color='green', linewidth=2, label="ROI (%)")
ax4.set_ylabel("ROI (%)", color='green')
ax4.tick_params(axis='y', labelcolor='green')

# Highlight Bear Market Periods
for start, end in bear_market_periods:
    ax3.axvspan(start, end, color='gray', alpha=0.3, label='Bear Market' if start == bear_market_periods[0][0] else "")

# Finalize the Plot
ax3.set_xlabel("Months")
ax3.set_title("ROI Evolution with Bear Markets and Overflow")

# Add legend
fig2.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
st.pyplot(fig2)

# Plot Unlocks Cumulative Across Scenarios
fig3, ax5 = plt.subplots(figsize=(12, 6))
ax5.bar(range(40), lower_unlocks_usd, label="Cumulative Unlocks (Lower Path)", color="green", alpha=0.5)
ax5.bar(range(40), median_unlocks_usd, label="Cumulative Unlocks (Median Path)", color="blue", alpha=0.5)
ax5.bar(range(40), upper_unlocks_usd, label="Cumulative Unlocks (Upper Path)", color="orange", alpha=0.5)

# Add Rewards Allocation to the Graph
ax5.bar(range(40), rewards_usd, bottom=median_unlocks_usd, color=allocations["Rewards"]["color"], label="Rewards", alpha=0.7)

# Overflow Hatching for Cumulative Unlocks
cumulative_excess = np.maximum(
    np.maximum(lower_unlocks_usd, median_unlocks_usd + rewards_usd),
    upper_unlocks_usd
) - dynamic_market_depth

ax5.bar(
    range(40),
    cumulative_excess.clip(min=0),
    bottom=dynamic_market_depth,
    color='none',
    edgecolor='red',
    hatch='//',
    label="Overflow"
)

ax5.step(range(40), dynamic_market_depth, where="mid", color="red", linestyle="--", label="Market Depth")
ax5.set_title("Cumulative Unlocks Across Scenarios")
ax5.set_xlabel("Months")
ax5.set_ylabel("Cumulative Unlocks (USD)")
ax5.legend()
st.pyplot(fig3)







# GPT Agent Integration for Overflow Analysis
st.write("### AI Agent Analysis for Overflow (Including Rewards)")

# Include Rewards in Overflow Calculation
rewards_usd = np.array(allocations["Rewards"]["unlock_schedule"]) * max_supply * median_path
adjusted_overflow = [
    max(0, selected_unlocks_usd[i] + rewards_usd[i] - dynamic_market_depth[i])
    for i in range(40)
]
adjusted_cumulative_overflow = sum(adjusted_overflow[range_start:range_end + 1])  # Cumulative overflow in the range

# Function to call the GPT model (fine-tuned or default)
import openai

openai.api_key = "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"  # Replace with your API key or use st.secrets["openai_api_key"]

def query_gpt(prompt, model="text-davinci-003", max_tokens=300, temperature=0.7):
    """
    Query GPT model (fine-tuned or default).
    
    Args:
        prompt (str): Input to the model.
        model (str): Model ID or name (e.g., "fine-tuned-model-id").
        max_tokens (int): Maximum number of tokens in the output.
        temperature (float): Creativity level of the model.

    Returns:
        str: Model-generated response.
    """
    try:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {e}"

# Overflow threshold
overflow_threshold = 250_000  # USD

if adjusted_cumulative_overflow > overflow_threshold:
    st.warning(f"Significant overflow detected in the selected range (including rewards): ${adjusted_cumulative_overflow:,.2f}")

    # Display choices
    st.write("#### Proposed Mitigation Strategies")
    strategy = st.radio(
        "Select a mitigation strategy:",
        [
            "1. Increase marketing efforts",
            "2. Add liquidity to the market",
            "3. Combine both marketing and liquidity",
        ],
    )

    # Detailed plan based on selection
    if st.button("Generate Strategy Plan"):
        if strategy == "1. Increase marketing efforts":
            prompt = f"""
            A significant token overflow of ${adjusted_cumulative_overflow:,.2f} has been detected in the selected range. 
            Develop a detailed marketing campaign plan to mitigate this issue, focusing on:
            - Enhancing brand visibility
            - Attracting more investors and token buyers
            - Reducing selling pressure through community engagement
            Provide a structured action plan with specific steps and expected outcomes.
            """
        elif strategy == "2. Add liquidity to the market":
            prompt = f"""
            A significant token overflow of ${adjusted_cumulative_overflow:,.2f} has been detected in the selected range.
            Develop a liquidity provisioning plan to stabilize the market, focusing on:
            - Increasing market depth by adding liquidity pools
            - Adjusting token unlock schedules
            - Minimizing price impact during high selling pressure
            Provide a structured action plan with clear steps and timelines.
            """
        else:
            prompt = f"""
            A significant token overflow of ${adjusted_cumulative_overflow:,.2f} has been detected in the selected range.
            Propose a combined strategy that includes:
            - Targeted marketing efforts to attract new buyers and retain existing ones
            - Liquidity provisioning to stabilize token price and reduce volatility
            Provide detailed steps for implementation, expected outcomes, and any potential risks.
            """

        # Query GPT model
        gpt_response = query_gpt(prompt, model="text-davinci-003")  # Replace with your fine-tuned model ID
        st.write("#### Suggested Plan of Action:")
        st.success(gpt_response)
else:
    st.info("No significant overflow detected in the selected range.")
