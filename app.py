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
show_data_points = st.sidebar.checkbox("Show Data Points", value=False)
show_ssr = st.sidebar.checkbox("Show Sum of Squared Residuals (SSR)", value=False)
show_ssr_vs_b = st.sidebar.checkbox("Show SSR vs. b plot", value=False)
show_ssr_vs_a = st.sidebar.checkbox("Show SSR vs. a plot", value=False)

# Generate scatter plot data with fixed parameters a=1, b=1 and Gaussian noise
# Set seed for reproducibility
np.random.seed(42)
scatter_x = np.linspace(-8, 8, 30)
scatter_y = 1 + 1 * scatter_x + np.random.normal(0, 1, size=len(scatter_x))

# Calculate Sum of Squared Residuals (SSR)
predicted_y = a + b * scatter_x
residuals = scatter_y - predicted_y
ssr = np.sum(residuals**2)

# Display the current equation with parameter values
st.markdown(f"<h4 style='text-align: center;'>y = <span style='color:red'>a</span> + <span style='color:green'>b</span> x</h4>", unsafe_allow_html=True)
st.markdown(f"<h4 style='text-align: center;'>y = <span style='color:red'>{a:.1f}</span> + <span style='color:green'>{b:.1f}</span> x</h4>", unsafe_allow_html=True)

# Display SSR value only if the show_ssr checkbox is checked
if show_ssr:
    st.markdown(f"<h4 style='text-align: center;'>Sum of Squared Residuals (SSR): <span style='color:blue'>{ssr:.2f}</span></h4>", unsafe_allow_html=True)

# Create plots - use columns if SSR vs. b plot or SSR vs. a plot is enabled
if show_ssr_vs_b and show_ssr_vs_a:
    # Show all three plots with equal width
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
            ax.scatter(scatter_x, scatter_y, color='red', alpha=0.7, label='Data Points (a=1, b=1, noise σ=1)')
            
            # Plot residuals as vertical lines if SSR is being shown
            if show_ssr:
                for i in range(len(scatter_x)):
                    ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], predicted_y[i]], 'g-', alpha=0.5)
        
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
    
    # Create the SSR vs. a plot in right column
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

elif show_ssr_vs_b:
    # Show main plot and SSR vs. b plot with equal width
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
            ax.scatter(scatter_x, scatter_y, color='red', alpha=0.7, label='Data Points (a=1, b=1, noise σ=1)')
            
            # Plot residuals as vertical lines if SSR is being shown
            if show_ssr:
                for i in range(len(scatter_x)):
                    ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], predicted_y[i]], 'g-', alpha=0.5)
        
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
    # Show main plot and SSR vs. a plot with equal width
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
            ax.scatter(scatter_x, scatter_y, color='red', alpha=0.7, label='Data Points (a=1, b=1, noise σ=1)')
            
            # Plot residuals as vertical lines if SSR is being shown
            if show_ssr:
                for i in range(len(scatter_x)):
                    ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], predicted_y[i]], 'g-', alpha=0.5)
        
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
else:
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
        ax.scatter(scatter_x, scatter_y, color='red', alpha=0.7, label='Data Points (a=1, b=1, noise σ=1)')
        
        # Plot residuals as vertical lines if SSR is being shown
        if show_ssr:
            for i in range(len(scatter_x)):
                ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], predicted_y[i]], 'g-', alpha=0.5)
    
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
