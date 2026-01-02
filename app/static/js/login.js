document.addEventListener("DOMContentLoaded", function () {
  const loginForm = document.getElementById("login-form");
  const errorMessage = document.getElementById("error-message");

  if (loginForm) {
    loginForm.addEventListener("submit", async function (e) {
      e.preventDefault();

      const usernameInput = document.getElementById("username");
      const passwordInput = document.getElementById("password");

      const username =
        usernameInput instanceof HTMLInputElement ? usernameInput.value : "";
      const password =
        passwordInput instanceof HTMLInputElement ? passwordInput.value : "";

      try {
        const response = await fetch("/api/token", {
          method: "POST",
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
          body: new URLSearchParams({
            username: username,
            password: password,
          }),
        });

        const data = await response.json();

        if (response.ok) {
          // Store token in localStorage
          localStorage.setItem("accessToken", data.access_token);

          // Redirect to dashboard
          window.location.href = "/";
        } else {
          if (errorMessage) {
            errorMessage.textContent =
              data.detail || "Login failed. Please check your credentials.";
          }
        }
      } catch (error) {
        console.error("Error:", error);
        if (errorMessage) {
          errorMessage.textContent =
            "An error occurred. Please try again later.";
        }
      }
    });
  }
});
