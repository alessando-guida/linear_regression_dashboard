import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set page title and description
st.markdown("<h1 style='text-align: center;'>Interactive Linear Regression Visualization</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Explore how parameters affect the linear regression model: </p>", unsafe_allow_html=True)

# Add a title to the sidebar
st.sidebar.header("Control Parameters")

# Create sliders for parameters a and b in the sidebar
a = st.sidebar.slider("Parameter a (y-intercept)", min_value=-10.0, max_value=10.0, value=-2.0, step=0.1)
b = st.sidebar.slider("Parameter b (slope)", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)

# Add checkbox to toggle the visibility of scatter plot data points and SSR
show_data_points = st.sidebar.checkbox("Show Train Set", value=False)
show_ssr = st.sidebar.checkbox("Show Sum of Squared Residuals (SSR)", value=False)
show_ssr_vs_b = st.sidebar.checkbox("Show SSR vs. b plot", value=False)
show_ssr_vs_a = st.sidebar.checkbox("Show SSR vs. a plot", value=False)

# Add checkbox to toggle the visibility of test set data points
show_test_set = st.sidebar.checkbox("Show Test Set Data", value=False)

# Add checkbox to toggle the visibility of evaluation metrics
eval_metrics = st.sidebar.checkbox("Show Evaluation Metrics", value=False)

# Add checkbox to toggle the visibility of MAE vs b plot
show_mae_vs_b = st.sidebar.checkbox("Show MAE vs. b plot", value=False)

# Generate scatter plot data with fixed parameters a=1, b=1 and Gaussian noise
# Set seed for reproducibility
np.random.seed(42)
scatter_x = np.linspace(-8, 8, 30)
scatter_y = 1 + 1 * scatter_x + np.random.normal(0, 1, size=len(scatter_x))

# Generate test set data with fixed parameters a=1, b=1 and Gaussian noise (std=2)
# Set seed for reproducibility
np.random.seed(24)
test_x = np.linspace(-8, 8, 30)
test_y = 1 + 1 * test_x + np.random.normal(0, 2, size=len(test_x))

# Calculate Sum of Squared Residuals (SSR)
predicted_y = a + b * scatter_x
residuals = scatter_y - predicted_y
ssr = np.sum(residuals**2)

# Calculate evaluation metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Calculate R-squared
r2_train = r2_score(scatter_y, predicted_y)
r2_test = r2_score(test_y, a + b * test_x)

# Calculate Adjusted R-squared
n_train = len(scatter_y)
p_train = 1  # number of predictors
adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p_train - 1)

n_test = len(test_y)
p_test = 1  # number of predictors
adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p_test - 1)

# Calculate Mean Absolute Error (MAE)
mae_train = mean_absolute_error(scatter_y, predicted_y)
mae_test = mean_absolute_error(test_y, a + b * test_x)

# Calculate Root Mean Squared Error (RMSE)
rmse_train = np.sqrt(mean_squared_error(scatter_y, predicted_y))
rmse_test = np.sqrt(mean_squared_error(test_y, a + b * test_x))

# Display the current equation with parameter values
st.markdown(f"<h4 style='text-align: center;'>y = <span style='color:red'>a</span> + <span style='color:green'>b</span> x</h4>", unsafe_allow_html=True)
st.markdown(f"<h4 style='text-align: center;'>y = <span style='color:red'>{a:.1f}</span> + <span style='color:green'>{b:.1f}</span> x</h4>", unsafe_allow_html=True)

# Display SSR value only if the show_ssr checkbox is checked
if show_ssr:
    st.markdown(f"<h4 style='text-align: center;'>Sum of Squared Residuals (SSR): <span style='color:blue'>{ssr:.2f}</span></h4>", unsafe_allow_html=True)


# Create plots - use columns if SSR vs. b plot, SSR vs. a plot, or MAE vs. b plot is enabled
if show_ssr_vs_b and show_ssr_vs_a and show_mae_vs_b:
    col1, col2, col3, col4 = st.columns(4)

    # Create the main regression plot in left column
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Generate x values for the line
        x = np.linspace(-10, 10, 100)
        
        # Calculate y values based on the linear equation
        y = a + b * x
        
        # Plot the line
        ax.plot(x, y, 'b-', linewidth=2, label='Interactive Line')
        
        # Plot the scatter points only if the checkbox is checked
        if show_data_points:
            ax.scatter(scatter_x, scatter_y, color='red', alpha=0.7, label='Train Set Points (a=1, b=1, noise σ=1)')
            
            # Plot residuals as vertical lines if SSR is being shown
            if show_ssr:
                for i in range(len(scatter_x)):
                    ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], predicted_y[i]], 'g-', alpha=0.5)
        
        # Plot the test set points only if the checkbox is checked
        if show_test_set:
            ax.scatter(test_x, test_y, color='purple', alpha=0.7, label='Test Set Points (a=1, b=1, noise σ=2)')
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set fixed axis limits
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        
        # Add labels and title
        plot_title = f'Linear Equation: y = {a:.1f} + {b:.1f}x'
        if show_ssr:
            plot_title += f' with SSR = {ssr:.2f}'
        ax.set_title(plot_title)
        
        # Add the x and y axes
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add a legend
        ax.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig)

    # Create the SSR vs. b plot in second column
    with col2:
        # Calculate SSR for different values of b
        b_range = np.linspace(-10, 10, 100)
        ssr_values = []
        
        # Current value of a
        current_a = a
        
        # Calculate SSR for each b value
        for b_val in b_range:
            pred_y = current_a + b_val * scatter_x
            res = scatter_y - pred_y
            ssr_values.append(np.sum(res**2))
        
        # Create the SSR vs. b plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(b_range, ssr_values, 'r-', linewidth=2)
        
        # Highlight the current b value
        ax2.axvline(x=b, color='g', linestyle='--', label=f'Current b = {b}')
        ax2.plot(b, ssr, 'go', markersize=8)
        
        # Add gridlines
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels and title
        ax2.set_xlabel('Parameter b (slope)')
        ax2.set_ylabel('Sum of Squared Residuals (SSR)')
        ax2.set_title('SSR vs. Parameter b (with fixed a)')
        
        # Add a legend
        ax2.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig2)

    # Create the SSR vs. a plot in third column
    with col3:
        # Calculate SSR for different values of a
        a_range = np.linspace(-10, 10, 100)
        ssr_values_a = []
        
        # Current value of b
        current_b = b
        
        # Calculate SSR for each a value
        for a_val in a_range:
            pred_y = a_val + current_b * scatter_x
            res = scatter_y - pred_y
            ssr_values_a.append(np.sum(res**2))
        
        # Create the SSR vs. a plot
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.plot(a_range, ssr_values_a, 'b-', linewidth=2)
        
        # Highlight the current a value
        ax3.axvline(x=a, color='r', linestyle='--', label=f'Current a = {a}')
        ax3.plot(a, ssr, 'ro', markersize=8)
        
        # Add gridlines
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels and title
        ax3.set_xlabel('Parameter a (y-intercept)')
        ax3.set_ylabel('Sum of Squared Residuals (SSR)')
        ax3.set_title('SSR vs. Parameter a (with fixed b)')
        
        # Add a legend
        ax3.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig3)

    # Create the MAE vs. b plot in fourth column
    with col4:
        # Calculate MAE for different values of b
        mae_values = []
        for b_val in b_range:
            pred_y = current_a + b_val * scatter_x
            mae_values.append(mean_absolute_error(scatter_y, pred_y))

        # Create the MAE vs. b plot
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        ax4.plot(b_range, mae_values, 'g-', linewidth=2)

        # Highlight the current b value
        ax4.axvline(x=b, color='r', linestyle='--', label=f'Current b = {b}')
        ax4.plot(b, mae_train, 'ro', markersize=8)

        # Add gridlines
        ax4.grid(True, linestyle='--', alpha=0.7)

        # Add labels and title
        ax4.set_xlabel('Parameter b (slope)')
        ax4.set_ylabel('Mean Absolute Error (MAE)')
        ax4.set_title('MAE vs. Parameter b (with fixed a)')

        # Add a legend
        ax4.legend()

        # Display the plot in Streamlit
        st.pyplot(fig4)

elif show_ssr_vs_b and show_mae_vs_b:
    col1, col2, col3 = st.columns(3)

    # Create the main regression plot in left column
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Generate x values for the line
        x = np.linspace(-10, 10, 100)
        
        # Calculate y values based on the linear equation
        y = a + b * x
        
        # Plot the line
        ax.plot(x, y, 'b-', linewidth=2, label='Interactive Line')
        
        # Plot the scatter points only if the checkbox is checked
        if show_data_points:
            ax.scatter(scatter_x, scatter_y, color='red', alpha=0.7, label='Train Set Points (a=1, b=1, noise σ=1)')
            
            # Plot residuals as vertical lines if SSR is being shown
            if show_ssr:
                for i in range(len(scatter_x)):
                    ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], predicted_y[i]], 'g-', alpha=0.5)
        
        # Plot the test set points only if the checkbox is checked
        if show_test_set:
            ax.scatter(test_x, test_y, color='purple', alpha=0.7, label='Test Set Points (a=1, b=1, noise σ=2)')
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set fixed axis limits
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        
        # Add labels and title
        plot_title = f'Linear Equation: y = {a:.1f} + {b:.1f}x'
        if show_ssr:
            plot_title += f' with SSR = {ssr:.2f}'
        ax.set_title(plot_title)
        
        # Add the x and y axes
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add a legend
        ax.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig)

    # Create the SSR vs. b plot in middle column
    with col2:
        # Calculate SSR for different values of b
        b_range = np.linspace(-10, 10, 100)
        ssr_values = []
        
        # Current value of a
        current_a = a
        
        # Calculate SSR for each b value
        for b_val in b_range:
            pred_y = current_a + b_val * scatter_x
            res = scatter_y - pred_y
            ssr_values.append(np.sum(res**2))
        
        # Create the SSR vs. b plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(b_range, ssr_values, 'r-', linewidth=2)
        
        # Highlight the current b value
        ax2.axvline(x=b, color='g', linestyle='--', label=f'Current b = {b}')
        ax2.plot(b, ssr, 'go', markersize=8)
        
        # Add gridlines
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels and title
        ax2.set_xlabel('Parameter b (slope)')
        ax2.set_ylabel('Sum of Squared Residuals (SSR)')
        ax2.set_title('SSR vs. Parameter b (with fixed a)')
        
        # Add a legend
        ax2.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig2)

    # Create the MAE vs. b plot in right column
    with col3:
        # Calculate MAE for different values of b
        mae_values = []
        for b_val in b_range:
            pred_y = current_a + b_val * scatter_x
            mae_values.append(mean_absolute_error(scatter_y, pred_y))

        # Create the MAE vs. b plot
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        ax4.plot(b_range, mae_values, 'g-', linewidth=2)

        # Highlight the current b value
        ax4.axvline(x=b, color='r', linestyle='--', label=f'Current b = {b}')
        ax4.plot(b, mae_train, 'ro', markersize=8)

        # Add gridlines
        ax4.grid(True, linestyle='--', alpha=0.7)

        # Add labels and title
        ax4.set_xlabel('Parameter b (slope)')
        ax4.set_ylabel('Mean Absolute Error (MAE)')
        ax4.set_title('MAE vs. Parameter b (with fixed a)')

        # Add a legend
        ax4.legend()

        # Display the plot in Streamlit
        st.pyplot(fig4)

elif show_ssr_vs_a and show_mae_vs_b:
    col1, col2, col3 = st.columns(3)

    # Create the main regression plot in left column
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Generate x values for the line
        x = np.linspace(-10, 10, 100)
        
        # Calculate y values based on the linear equation
        y = a + b * x
        
        # Plot the line
        ax.plot(x, y, 'b-', linewidth=2, label='Interactive Line')
        
        # Plot the scatter points only if the checkbox is checked
        if show_data_points:
            ax.scatter(scatter_x, scatter_y, color='red', alpha=0.7, label='Train Set Points (a=1, b=1, noise σ=1)')
            
            # Plot residuals as vertical lines if SSR is being shown
            if show_ssr:
                for i in range(len(scatter_x)):
                    ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], predicted_y[i]], 'g-', alpha=0.5)
        
        # Plot the test set points only if the checkbox is checked
        if show_test_set:
            ax.scatter(test_x, test_y, color='purple', alpha=0.7, label='Test Set Points (a=1, b=1, noise σ=2)')
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set fixed axis limits
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        
        # Add labels and title
        plot_title = f'Linear Equation: y = {a:.1f} + {b:.1f}x'
        if show_ssr:
            plot_title += f' with SSR = {ssr:.2f}'
        ax.set_title(plot_title)
        
        # Add the x and y axes
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add a legend
        ax.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig)

    # Create the SSR vs. a plot in middle column
    with col2:
        # Calculate SSR for different values of a
        a_range = np.linspace(-10, 10, 100)
        ssr_values_a = []
        
        # Current value of b
        current_b = b
        
        # Calculate SSR for each a value
        for a_val in a_range:
            pred_y = a_val + current_b * scatter_x
            res = scatter_y - pred_y
            ssr_values_a.append(np.sum(res**2))
        
        # Create the SSR vs. a plot
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.plot(a_range, ssr_values_a, 'b-', linewidth=2)
        
        # Highlight the current a value
        ax3.axvline(x=a, color='r', linestyle='--', label=f'Current a = {a}')
        ax3.plot(a, ssr, 'ro', markersize=8)
        
        # Add gridlines
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels and title
        ax3.set_xlabel('Parameter a (y-intercept)')
        ax3.set_ylabel('Sum of Squared Residuals (SSR)')
        ax3.set_title('SSR vs. Parameter a (with fixed b)')
        
        # Add a legend
        ax3.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig3)

    # Create the MAE vs. b plot in right column
    with col3:
        # Calculate MAE for different values of b
        mae_values = []
        for b_val in b_range:
            pred_y = current_a + b_val * scatter_x
            mae_values.append(mean_absolute_error(scatter_y, pred_y))

        # Create the MAE vs. b plot
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        ax4.plot(b_range, mae_values, 'g-', linewidth=2)

        # Highlight the current b value
        ax4.axvline(x=b, color='r', linestyle='--', label=f'Current b = {b}')
        ax4.plot(b, mae_train, 'ro', markersize=8)

        # Add gridlines
        ax4.grid(True, linestyle='--', alpha=0.7)

        # Add labels and title
        ax4.set_xlabel('Parameter b (slope)')
        ax4.set_ylabel('Mean Absolute Error (MAE)')
        ax4.set_title('MAE vs. Parameter b (with fixed a)')

        # Add a legend
        ax4.legend()

        # Display the plot in Streamlit
        st.pyplot(fig4)

elif show_ssr_vs_b:
    col1, col2 = st.columns(2)

    # Create the main regression plot in left column
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Generate x values for the line
        x = np.linspace(-10, 10, 100)
        
        # Calculate y values based on the linear equation
        y = a + b * x
        
        # Plot the line
        ax.plot(x, y, 'b-', linewidth=2, label='Interactive Line')
        
        # Plot the scatter points only if the checkbox is checked
        if show_data_points:
            ax.scatter(scatter_x, scatter_y, color='red', alpha=0.7, label='Train set Points (a=1, b=1, noise σ=1)')
            
            # Plot residuals as vertical lines if SSR is being shown
            if show_ssr:
                for i in range(len(scatter_x)):
                    ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], predicted_y[i]], 'g-', alpha=0.5)
        
        # Plot the test set points only if the checkbox is checked
        if show_test_set:
            ax.scatter(test_x, test_y, color='purple', alpha=0.7, label='Test Set Points (a=1, b=1, noise σ=2)')
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set fixed axis limits
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        
        # Add labels and title
        plot_title = f'Linear Equation: y = {a:.1f} + {b:.1f}x'
        if show_ssr:
            plot_title += f' with SSR = {ssr:.2f}'
        ax.set_title(plot_title)
        
        # Add the x and y axes
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add a legend
        ax.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig)

    # Create the SSR vs. b plot in right column
    with col2:
        # Calculate SSR for different values of b
        b_range = np.linspace(-10, 10, 100)
        ssr_values = []
        
        # Current value of a
        current_a = a
        
        # Calculate SSR for each b value
        for b_val in b_range:
            pred_y = current_a + b_val * scatter_x
            res = scatter_y - pred_y
            ssr_values.append(np.sum(res**2))
        
        # Create the SSR vs. b plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(b_range, ssr_values, 'r-', linewidth=2)
        
        # Highlight the current b value
        ax2.axvline(x=b, color='g', linestyle='--', label=f'Current b = {b}')
        ax2.plot(b, ssr, 'go', markersize=8)
        
        # Add gridlines
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels and title
        ax2.set_xlabel('Parameter b (slope)')
        ax2.set_ylabel('Sum of Squared Residuals (SSR)')
        ax2.set_title('SSR vs. Parameter b (with fixed a)')
        
        # Add a legend
        ax2.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig2)

elif show_ssr_vs_a:
    col1, col2 = st.columns(2)

    # Create the main regression plot in left column
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Generate x values for the line
        x = np.linspace(-10, 10, 100)
        
        # Calculate y values based on the linear equation
        y = a + b * x
        
        # Plot the line
        ax.plot(x, y, 'b-', linewidth=2, label='Interactive Line')
        
        # Plot the scatter points only if the checkbox is checked
        if show_data_points:
            ax.scatter(scatter_x, scatter_y, color='red', alpha=0.7, label='Train set Points (a=1, b=1, noise σ=1)')
            
            # Plot residuals as vertical lines if SSR is being shown
            if show_ssr:
                for i in range(len(scatter_x)):
                    ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], predicted_y[i]], 'g-', alpha=0.5)
        
        # Plot the test set points only if the checkbox is checked
        if show_test_set:
            ax.scatter(test_x, test_y, color='Purple', alpha=0.7, label='Test Set Points (a=1, b=1, noise σ=2)')
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set fixed axis limits
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        
        # Add labels and title
        plot_title = f'Linear Equation: y = {a:.1f} + {b:.1f}x'
        if show_ssr:
            plot_title += f' with SSR = {ssr:.2f}'
        ax.set_title(plot_title)
        
        # Add the x and y axes
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add a legend
        ax.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig)

    # Create the SSR vs. a plot in right column
    with col2:
        # Calculate SSR for different values of a
        a_range = np.linspace(-10, 10, 100)
        ssr_values_a = []
        
        # Current value of b
        current_b = b
        
        # Calculate SSR for each a value
        for a_val in a_range:
            pred_y = a_val + current_b * scatter_x
            res = scatter_y - pred_y
            ssr_values_a.append(np.sum(res**2))
        
        # Create the SSR vs. a plot
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.plot(a_range, ssr_values_a, 'b-', linewidth=2)
        
        # Highlight the current a value
        ax3.axvline(x=a, color='r', linestyle='--', label=f'Current a = {a}')
        ax3.plot(a, ssr, 'ro', markersize=8)
        
        # Add gridlines
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels and title
        ax3.set_xlabel('Parameter a (y-intercept)')
        ax3.set_ylabel('Sum of Squared Residuals (SSR)')
        ax3.set_title('SSR vs. Parameter a (with fixed b)')
        
        # Add a legend
        ax3.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig3)

elif show_mae_vs_b:
    col1, col2 = st.columns(2)

    # Create the main regression plot in left column
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Generate x values for the line
        x = np.linspace(-10, 10, 100)
        
        # Calculate y values based on the linear equation
        y = a + b * x
        
        # Plot the line
        ax.plot(x, y, 'b-', linewidth=2, label='Interactive Line')
        
        # Plot the scatter points only if the checkbox is checked
        if show_data_points:
            ax.scatter(scatter_x, scatter_y, color='red', alpha=0.7, label='Train set Points (a=1, b=1, noise σ=1)')
            
            # Plot residuals as vertical lines if SSR is being shown
            if show_ssr:
                for i in range(len(scatter_x)):
                    ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], predicted_y[i]], 'g-', alpha=0.5)
        
        # Plot the test set points only if the checkbox is checked
        if show_test_set:
            ax.scatter(test_x, test_y, color='purple', alpha=0.7, label='Test Set Points (a=1, b=1, noise σ=2)')
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set fixed axis limits
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        
        # Add labels and title
        plot_title = f'Linear Equation: y = {a:.1f} + {b:.1f}x'
        if show_ssr:
            plot_title += f' with SSR = {ssr:.2f}'
        ax.set_title(plot_title)
        
        # Add the x and y axes
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add a legend
        ax.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig)

    # Create the MAE vs. b plot in right column
    with col2:
        # Calculate MAE for different values of b
        b_range = np.linspace(-10, 10, 100)
        mae_values = []
        current_a = a
        for b_val in b_range:
            pred_y = current_a + b_val * scatter_x
            mae_values.append(mean_absolute_error(scatter_y, pred_y))

        # Create the MAE vs. b plot
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        ax4.plot(b_range, mae_values, 'g-', linewidth=2)

        # Highlight the current b value
        ax4.axvline(x=b, color='r', linestyle='--', label=f'Current b = {b}')
        ax4.plot(b, mae_train, 'ro', markersize=8)

        # Add gridlines
        ax4.grid(True, linestyle='--', alpha=0.7)

        # Add labels and title
        ax4.set_xlabel('Parameter b (slope)')
        ax4.set_ylabel('Mean Absolute Error (MAE)')
        ax4.set_title('MAE vs. Parameter b (with fixed a)')

        # Add a legend
        ax4.legend()

        # Display the plot in Streamlit
        st.pyplot(fig4)
else:
    col1 = st.columns(1)[0]
    # Create just the main regression plot without columns
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate x values for the line
    x = np.linspace(-10, 10, 100)
    
    # Calculate y values based on the linear equation
    y = a + b * x
    
    # Plot the line
    ax.plot(x, y, 'b-', linewidth=2, label='Interactive Line')
    
    # Plot the scatter points only if the checkbox is checked
    if show_data_points:
        ax.scatter(scatter_x, scatter_y, color='red', alpha=0.7, label='Train set Points (a=1, b=1, noise σ=1)')
        
        # Plot residuals as vertical lines if SSR is being shown
        if show_ssr:
            for i in range(len(scatter_x)):
                ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], predicted_y[i]], 'g-', alpha=0.5)
    
    # Plot the test set points only if the checkbox is checked
    if show_test_set:
        ax.scatter(test_x, test_y, color='purple', alpha=0.7, label='Test Set Points (a=1, b=1, noise σ=2)')
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set fixed axis limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    
    # Add labels and title
    plot_title = f'Linear Equation: y = {a:.1f} + {b:.1f}x'
    if show_ssr:
        plot_title += f' with SSR = {ssr:.2f}'
    ax.set_title(plot_title)
    
    # Add the x and y axes
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add a legend
    ax.legend()
    
    # Display the plot in Streamlit
    st.pyplot(fig)


# Display evaluation metrics only if the eval_metrics checkbox is checked
if eval_metrics:
    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown("<h4 style='text-align: center;'>Evaluation Metrics</h4>", unsafe_allow_html=True)
        st.markdown("<table style='width:100%; text-align: center;'>"
                    "<tr><th></th><th>Train Set</th><th>Test Set</th></tr>"
                    f"<tr><td>R-squared</td><td>{r2_train:.2f}</td><td>{r2_test:.2f}</td></tr>"
                    f"<tr><td>Adjusted R-squared</td><td>{adj_r2_train:.2f}</td><td>{adj_r2_test:.2f}</td></tr>"
                    f"<tr><td>MAE</td><td>{mae_train:.2f}</td><td>{mae_test:.2f}</td></tr>"
                    f"<tr><td>RMSE</td><td>{rmse_train:.2f}</td><td>{rmse_test:.2f}</td></tr>"
                    "</table>", unsafe_allow_html=True)
    with col1:
        # Create the main regression plot in left column
        fig, ax = plt.subplots(figsize=(8, 6))
        # Generate x values for the line
        x = np.linspace(-10, 10, 100)
        
        # Calculate y values based on the linear equation
        y = a + b * x
        
        # Plot the line
        ax.plot(x, y, 'b-', linewidth=2, label='Interactive Line')
        
        # Plot the scatter points only if the checkbox is checked
        if show_data_points:
            ax.scatter(scatter_x, scatter_y, color='red', alpha=0.7, label='Train Set Points (a=1, b=1, noise σ=1)')
            
            # Plot residuals as vertical lines if SSR is being shown
            if show_ssr:
                for i in range(len(scatter_x)):
                    ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], predicted_y[i]], 'g-', alpha=0.5)
        
        # Plot the test set points only if the checkbox is checked
        if show_test_set:
            ax.scatter(test_x, test_y, color='purple', alpha=0.7, label='Test Set Points (a=1, b=1, noise σ=2)')
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set fixed axis limits
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        
        # Add labels and title
        plot_title = f'Linear Equation: y = {a:.1f} + {b:.1f}x'
        if show_ssr:
            plot_title += f' with SSR = {ssr:.2f}'
        ax.set_title(plot_title)
        
        # Add the x and y axes
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add a legend
        ax.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig)
else:
    col1 = st.columns(1)[0]