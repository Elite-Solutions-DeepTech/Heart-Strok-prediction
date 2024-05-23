{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "L_rxvbAVAXlE",
        "outputId": "bd8f8e5e-ba2a-441f-8bb3-b7cc7f7fc9cd"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "          ID  Gender   Age  Hypertension  Heart_Disease Ever_Married  \\\n",
            "0      30669    Male   3.0             0              0           No   \n",
            "1      30468    Male  58.0             1              0          Yes   \n",
            "2      16523  Female   8.0             0              0           No   \n",
            "3      56543  Female  70.0             0              0          Yes   \n",
            "4      46136    Male  14.0             0              0           No   \n",
            "...      ...     ...   ...           ...            ...          ...   \n",
            "43395  56196  Female  10.0             0              0           No   \n",
            "43396   5450  Female  56.0             0              0          Yes   \n",
            "43397  28375  Female  82.0             1              0          Yes   \n",
            "43398  27973    Male  40.0             0              0          Yes   \n",
            "43399  36271  Female  82.0             0              0          Yes   \n",
            "\n",
            "          Work_Type Residence_Type  Avg_Glucose_Level   BMI   Smoking_Status  \\\n",
            "0          children          Rural              95.12  18.0              NaN   \n",
            "1           Private          Urban              87.96  39.2     never smoked   \n",
            "2           Private          Urban             110.89  17.6              NaN   \n",
            "3           Private          Rural              69.04  35.9  formerly smoked   \n",
            "4      Never_worked          Rural             161.28  19.1              NaN   \n",
            "...             ...            ...                ...   ...              ...   \n",
            "43395      children          Urban              58.64  20.4     never smoked   \n",
            "43396      Govt_job          Urban             213.61  55.4  formerly smoked   \n",
            "43397       Private          Urban              91.94  28.9  formerly smoked   \n",
            "43398       Private          Urban              99.16  33.2     never smoked   \n",
            "43399       Private          Urban              79.48  20.6     never smoked   \n",
            "\n",
            "       Stroke  \n",
            "0           0  \n",
            "1           0  \n",
            "2           0  \n",
            "3           0  \n",
            "4           0  \n",
            "...       ...  \n",
            "43395       0  \n",
            "43396       0  \n",
            "43397       0  \n",
            "43398       0  \n",
            "43399       0  \n",
            "\n",
            "[43400 rows x 12 columns]\n",
            "        Age\n",
            "0       3.0\n",
            "1      58.0\n",
            "2       8.0\n",
            "3      70.0\n",
            "4      14.0\n",
            "...     ...\n",
            "43395  10.0\n",
            "43396  56.0\n",
            "43397  82.0\n",
            "43398  40.0\n",
            "43399  82.0\n",
            "\n",
            "[43400 rows x 1 columns]\n",
            "0        0\n",
            "1        0\n",
            "2        0\n",
            "3        0\n",
            "4        0\n",
            "        ..\n",
            "43395    0\n",
            "43396    0\n",
            "43397    0\n",
            "43398    0\n",
            "43399    0\n",
            "Name: Stroke, Length: 43400, dtype: int64\n",
            "(34720, 1)\n",
            "(34720,)\n",
            "(8680, 1)\n",
            "(8680,)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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ICAAgYGBuPXWW/HYY49h37598Pf3x/jx4819Dh48iPvvvx9+fn7Q6XTo168fDh8+bLGNADBo0CDIZDLzc3v2DxERNU4MN062detWDBkyBJdVegQ++TaCn9uEwCffxmWVHkOGDKm3gFOTmTNn4o033sDJkyfRuXNn5OXloX///oiLi8ORI0fw4IMPYsCAAUhKSqp2nFdffRVDhw7FX3/9hf79+2P48OG4cuWKzf4FBQV4++238emnn+KXX35BUlISpk+fbn79zTffxGeffYa1a9di3759yMnJqfVHRO7u7hg3bhz27duHjIwMAEBubi5GjRqFvXv34vfff0d4eDj69++P3NxcAGXhByj7q+apqanm57XdP0RE1AiIm4zBYBAAhMFgqPJaYWGhOHHihCgsLKzV2KWlpSK4dYjwuCVCtH5xuwiZscP8aP3iduFxS4QIDgkVpaWldd0Mm9auXSt0Op35+e7duwUAsW3bthqXvf3228W7775rfh4SEiKWLl1qfg5AzJ492/w8Ly9PABDfffedxbquXr1qrgWAOHv2rHmZFStWCL1eb36u1+vFokWLzM9LS0tF69atxcCBA23WWXk9FX333XcCgNi/f7/VZY1Go/Dy8hJff/21xXZ9+eWXNtdXrvL+caa6HntERFJX3e/vynjmxol+/fVXJCddhPedQyGTWe5amUwO7zsfRfLFC/j1118bvLYePXpYPM/Ly8P06dNx2223wcfHB1qtFidPnqzxzETnzp3N//b09IS3t7f5LIk1Hh4eaNu2rfl5UFCQub/BYEB6ejoiIiLMrysUCnTv3t2hbatICAGg7CM0AEhPT8fYsWMRHh4OnU4Hb29v5OXl1bidtd0/RETkeryg2InKryVx8w+x+rqbX4hFv4bk6elp8Xz69OnYtWsX3n77bdxyyy1wd3fHkCFDUFxcXO04bm5uFs9lMhlMJpND/csDSH04efIkgBvX0owaNQqXL1/Gf//7X4SEhECtViMyMrLG7azt/iEiItdjuHGioKAgAEBJ5kWoW7av8npJ1kWLfq60b98+jB49GoMGDQJQdqbiwoULDVqDTqeDXq/HwYMHcffddwMo+5bZ4cOH0bVrV4fHKywsxOrVq3H33Xebv+m1b98+vPfee+jfvz8AIDk5GVlZWRbLubm5wWg0WrQ1hv1DRES1w4+lnKhv374Ibh2C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jURERER2czjcnDx5EiNHjgQAKJVKFBYWQqvV4rXXXsObb77p9AKJiIiIHOFwuPH09DRfZxMUFIRz586ZX8vKynJeZURERES1YHe4ee2115Cfn48777wTe/fuBQD0798fzz//PBYsWICnnnoKd955Z70VSkRERGQPu2/ip1AokJqairy8POTl5aFz587Iz8/H888/j99++w3h4eFYsmQJQkJC6rvmOuFN/IiIiJqeermJX3kGatOmjbnN09MTq1atqmWZRERERM7n0DU3MpmsvuogIiIicgqH7nNz66231hhwrly5UqeCiIiIiOrCoXDz6quvQqfT1VctRERERHXmULh57LHHEBAQUF+1EBEREdWZ3dfc8HobIiIiagrsDjd2fmOciIiIyKXs/ljKZDLVZx1ERERETuHwn18gIiIiaswYboiIiEhSGkW4WbFiBUJDQ6HRaNCrVy8cOHDAruU2bNgAmUyGmJiY+i2QiIiImgyXh5uNGzdi2rRpmDdvHg4fPowuXbogOjoaGRkZ1S534cIFTJ8+HX379m2gSomIiKgpcHm4WbJkCcaOHYsxY8agQ4cOWLVqFTw8PPDRRx/ZXMZoNGL48OF49dVXLf7WlTVFRUXIycmxeBAREZF0uTTcFBcX49ChQ4iKijK3yeVyREVFIT4+3uZyr732GgICAvCf//ynxnXExsZCp9OZH8HBwU6pnYiIiBonl4abrKwsGI1G6PV6i3a9Xo+0tDSry+zduxdr1qzBBx98YNc6Zs2aBYPBYH4kJyfXuW4iIiJqvBz68wuulpubixEjRuCDDz6An5+fXcuo1Wqo1ep6royIiIgaC5eGGz8/PygUCqSnp1u0p6enIzAwsEr/c+fO4cKFCxgwYIC5rfzmgkqlEgkJCWjbtm39Fk1ERESNmks/llKpVOjevTvi4uLMbSaTCXFxcYiMjKzSv3379jh27BiOHj1qfvz73//Gvffei6NHj/J6GiIiInL9x1LTpk3DqFGj0KNHD0RERGDZsmXIz8/HmDFjAAAjR45Ey5YtERsbC41Gg44dO1os7+PjAwBV2omIiOjm5PJwM2zYMGRmZmLu3LlIS0tD165dsXPnTvNFxklJSZDLXf6NdSIiImoiZOIm+3PfOTk50Ol0MBgM8Pb2dnU5REREZAdHfn/zlAgRERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJSqMINytWrEBoaCg0Gg169eqFAwcO2Oz7wQcfoG/fvmjWrBmaNWuGqKioavsTERHRzcXl4Wbjxo2YNm0a5s2bh8OHD6NLly6Ijo5GRkaG1f579uzB448/jt27dyM+Ph7BwcF44IEHkJKS0sCVExERUWMkE0IIVxbQq1cv9OzZE8uXLwcAmEwmBAcHY9KkSZg5c2aNyxuNRjRr1gzLly/HyJEjq7xeVFSEoqIi8/OcnBwEBwfDYDDA29vbeRtCRERE9SYnJwc6nc6u398uPXNTXFyMQ4cOISoqytwml8sRFRWF+Ph4u8YoKChASUkJmjdvbvX12NhY6HQ68yM4ONgptRMREVHj5NJwk5WVBaPRCL1eb9Gu1+uRlpZm1xgzZsxAixYtLAJSRbNmzYLBYDA/kpOT61w3ERERNV5KVxdQF2+88QY2bNiAPXv2QKPRWO2jVquhVqsbuDIiIiJyFZeGGz8/PygUCqSnp1u0p6enIzAwsNpl3377bbzxxhv48ccf0blz5/osk4iIiJoQl34spVKp0L17d8TFxZnbTCYT4uLiEBkZaXO5t956C/Pnz8fOnTvRo0ePhiiViIiImgiXfyw1bdo0jBo1Cj169EBERASWLVuG/Px8jBkzBgAwcuRItGzZErGxsQCAN998E3PnzsX69esRGhpqvjZHq9VCq9W6bDuIiIiocXB5uBk2bBgyMzMxd+5cpKWloWvX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1WLw4MFIT093UcVU2RtvvAGZTIapU6ea2zhnjVdKSgqefPJJ+Pr6wt3dHZ06dcIff/xhfl0Igblz5yIoKAju7u6IiorCmTNnXFgxGY1GzJkzB2FhYXB3d0fbtm0xf/58i7/l1GTmTVCdbdiwQahUKvHRRx+Jv//+W4wdO1b4+PiI9PR0V5dG10VHR4u1a9eK48ePi6NHj4r+/fuL1q1bi7y8PHOfcePGieDgYBEXFyf++OMPceedd4revXu7sGoqd+DAAREaGio6d+4spkyZYm7nnDVOV65cESEhIWL06NFi//794vz58+L7778XZ8+eNfd54403hE6nE9u2bRN//vmn+Pe//y3CwsJEYWGhCyu/uS1YsED4+vqKHTt2iMTERLFp0yah1WrFf//7X3OfpjJvDDdOEBERISZMmGB+bjQaRYsWLURsbKwLq6LqZGRkCADi559/FkIIkZ2dLdzc3MSmTZvMfU6ePCkAiPj4eFeVSUKI3NxcER4eLnbt2iX69etnDjecs8ZrxowZ4q677rL5uslkEoGBgWLRokXmtuzsbKFWq8Xnn3/eECWSFQ899JB46qmnLNoeeeQRMXz4cCFE05o3fixVR8XFxTh06BCioqLMbXK5HFFRUYiPj3dhZVQdg8EAAGjevDkA4NChQygpKbGYx/bt26N169acRxebMGECHnroIYu5AThnjdn27dvRo0cPPProowgICEC3bt3wwQcfmF9PTExEWlqaxdzpdDr06tWLc+dCvXv3RlxcHE6fPg0A+PPPP7F3717861//AtC05u2m+8OZzpaVlQWj0Qi9Xm/RrtfrcerUKRdVRdUxmUyYOnUq+vTpg44dOwIA0tLSoFKp4OPjY9FXr9cjLS3NBVUSAGzYsAGHDx/GwYMHq7zGOWu8zp8/j5UrV2LatGl46aWXcPDgQUyePBkqlQqjRo0yz4+1n5ucO9eZOXMmcnJy0L59eygUChiNRixYsADDhw8HgCY1bww3dNOZMGECjh8/jr1797q6FKpGcnIypkyZgl27dkGj0bi6HHKAyWRCjx49sHDhQgBAt27dcPz4caxatQqjRo1ycXVkyxdffIHPPvsM69evx+23346jR49i6tSpaNGiRZObN34sVUd+fn5QKBRVvqGRnp6OwMBAF1VFtkycOBE7duzA7t270apVK3N7YGAgiouLkZ2dbdGf8+g6hw4dQkZGBu644w4olUoolUr8/PPPeOedd6BUKqHX6zlnjVRQUBA6dOhg0XbbbbchKSkJAMzzw5+bjcsLL7yAmTNn4rHHHkOnTp0wYsQIPPfcc4iNjQXQtOaN4aaOVCoVunfvjri4OHObyWRCXFwcIiMjXVgZVSSEwMSJE/Hll1/ip59+QlhYmMXr3bt3h5ubm8U8JiQkICkpifPoIvfddx+OHTuGo0ePmh89evTA8OHDzf/mnDVOffr0qXKrhdOnTyMkJAQAEBYWhsDAQIu5y8nJwf79+zl3LlRQUAC53DIWKBQKmEwmAE1s3lx9RbMUbNiwQajVarFu3Tpx4sQJ8cwzzwgfHx+Rlpbm6tLouvHjxwudTif27NkjUlNTzY+CggJzn3HjxonWrVuLn376Sfzxxx8iMjJSREZGurBqqqzit6WE4Jw1VgcOHBBKpVIsWLBAnDlzRnz22WfCw8ND/O9//zP3eeONN4SPj4/46quvxF9//SUGDhzYKL9SfDMZNWqUaNmypfmr4Fu3bhV+fn7ixRdfNPdpKvPGcOMk7777rmjdurVQqVQiIiJC/P77764uiSoAYPWxdu1ac5/CwkLx//7f/xPNmjUTHh4eYtCgQSI1NdV1RVMVlcMN56zx+vrrr0XHjh2FWq0W7du3F6tXr7Z43WQyiTlz5gi9Xi/UarW47777REJCgouqJSGEyMnJEVOmTBGtW7cWGo1GtGnTRrz88suiqKjI3KepzJtMiAq3HiQiIiJq4njNDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDREREUkKww0RERFJCsMNERERSQrDDRE1CfHx8VAoFHjooYdcXQoRNXL88wtE1CQ8/fTT0Gq1WLNmDRISEtCiRQtXl0REjRTP3BBRo5eXl4eNGzdi/PjxeOihh7Bu3TqL17dv347w8HBoNBrce++9+PjjjyGTyZCdnW3us3fvXvTt2xfu7u4IDg7G5MmTkZ+f37AbQkQNguGGiBq9L774Au3bt0e7du3w5JNP4qOPPkL5SefExEQMGTIEMTEx+PPPP/Hss8/i5Zdftlj+3LlzePDBBzF48GD89ddf2LhxI/bu3YuJEye6YnOIqJ7xYykiavT69OmDoUOHYsqUKSgtLUVQUBA2bdqEe+65BzNnzsQ333yDY8eOmfvPnj0bCxYswNWrV+Hj44Onn34aCoUC77//vrnP3r170a9fP+Tn50Oj0bhis4ionvDMDRE1agkJCThw4AAef/xxAIBSqcSwYcOwZs0a8+s9e/a0WCYiIsLi+Z9//ol169ZBq9WaH9HR0TCZTEhMTGyYDSGiBqN0dQFERNVZs2YNSktLLS4gFkJArVZj+fLldo2Rl5eHZ599FpMnT67yWuvWrZ1WKxE1Dgw3RNRolZaW4pNPPsHixYvxwAMPWLwWExODzz//HO3atcO3335r8drBgwctnt9xxx04ceIEbrnllnqvmYhcj9fcEFGjtW3bNgwbNgwZGRnQ6XQWr82YMQM//fQTvvjiC7Rr1w7PPfcc/vOf/+Do0aN4/vnncenSJWRnZ0On0+Gvv/7CnXfeiaeeegpPP/00PD09ceLECezatcvusz9E1HTwmhsiarTWrFmDqKioKsEGAAYPHow//vgDubm52Lx5M7Zu3YrOnTtj5cqV5m9LqdVqAEDnzp3x888/4/Tp0+jbty+6deuGuXPn8l45RBLFMzdEJDkLFizAqlWrkJyc7OpSiMgFeM0NETV57733Hnr27AlfX1/s27cPixYt4j1siG5iDDdE1OSdOXMGr7/+Oq5cuYLWrVvj+eefx6xZs1xdFhG5CD+WIiIiIknhBcVEREQkKQw3REREJCkMN0RERCQpDDdEREQkKQw3REREJCkMN0RERCQpDDdEREQkKQw3REREJCn/H7ErR2ZIkY6RAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Score: 0.024939035984545943\n",
            "Testing Score: 0.02205508955021862\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.linear_model import LinearRegression\n",
        "\n",
        "\n",
        "variable = pd.read_csv('/content/Stroke.csv')\n",
        "print(variable)\n",
        "\n",
        "\n",
        "x = variable[['Age']]\n",
        "y = variable.iloc[:, -1]\n",
        "print(x)\n",
        "print(y)\n",
        "\n",
        "\n",
        "imputer = SimpleImputer(strategy='mean')\n",
        "x_imputed = imputer.fit_transform(x)\n",
        "\n",
        "\n",
        "x_train, x_test, y_train, y_test = train_test_split(x_imputed, y, test_size=0.2, random_state=42)\n",
        "print(x_train.shape)\n",
        "print(y_train.shape)\n",
        "print(x_test.shape)\n",
        "print(y_test.shape)\n",
        "\n",
        "\n",
        "alg = LinearRegression()\n",
        "alg.fit(x_train, y_train)\n",
        "\n",
        "\n",
        "m = alg.coef_[0]\n",
        "c = alg.intercept_\n",
        "\n",
        "\n",
        "x_line = np.arange(x_train.min(), x_train.max(), 0.1).reshape(-1, 1)\n",
        "y_line = alg.predict(x_line)\n",
        "\n",
        "\n",
        "plt.plot(x_line, y_line, label='Regression Line')\n",
        "plt.scatter(x_train, y_train, label='Training Data', edgecolors='k', marker='o')\n",
        "plt.xlabel('Age')\n",
        "plt.ylabel('Target')\n",
        "plt.legend()\n",
        "plt.show()\n",
        "\n",
        "\n",
        "print('Training Score:', alg.score(x_train, y_train))\n",
        "print('Testing Score:', alg.score(x_test, y_test))\n"
      ]
    }
  ]
}
