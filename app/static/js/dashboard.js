document.addEventListener("DOMContentLoaded", function () {
  const queryForm = document.getElementById("query-form");
  const responseContent = document.getElementById("response-content");
  const logoutBtn = document.getElementById("logout-btn");
  const userName = document.getElementById("user-name");

  const token = localStorage.getItem("accessToken");

  if (token) {
    fetchUserInfo(token);
  } else {
    if (userName) userName.textContent = "Guest";
    if (logoutBtn) {
      logoutBtn.textContent = "Login";
      logoutBtn.setAttribute("href", "/login");
    }
  }

  if (logoutBtn) {
    logoutBtn.addEventListener("click", function (e) {
      e.preventDefault();
      if (token) {
        localStorage.removeItem("accessToken");
        window.location.href = "/";
      } else {
        window.location.href = "/login";
      }
    });
  }

  async function fetchUserInfo(token) {
    try {
      const response = await fetch("/api/users/me", {
        method: "GET",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const userData = await response.json();
        if (userName)
          userName.textContent = userData.full_name || userData.username;
      } else {
        if (response.status === 401) {
          localStorage.removeItem("accessToken");
          if (userName) userName.textContent = "Guest";
          if (logoutBtn) logoutBtn.textContent = "Login";
        }
      }
    } catch (error) {
      console.error("Error fetching user info:", error);
      if (userName) userName.textContent = "Guest";
      if (logoutBtn) logoutBtn.textContent = "Login";
    }
  }

  if (queryForm) {
    queryForm.addEventListener("submit", async function (e) {
      e.preventDefault();

      let query = "";
      const queryElement = document.getElementById("query");
      if (
        queryElement instanceof HTMLInputElement ||
        queryElement instanceof HTMLTextAreaElement
      ) {
        query = queryElement.value;
      }

      if (responseContent) {
        responseContent.innerHTML =
          '<p class="placeholder">Processing your query...</p>';
      }

      try {
        const headers = {
          "Content-Type": "application/json",
        };

        if (token) {
          headers.Authorization = `Bearer ${token}`;
        }

        const response = await fetch("/api/secure-query", {
          method: "POST",
          headers: headers,
          body: JSON.stringify({
            query: query,
          }),
        });

        const data = await response.json();

        if (response.ok && responseContent) {
          let responseHTML = `<div class="ai-response">${formatResponse(
            data.response
          )}</div>`;

          responseContent.innerHTML = responseHTML;
        } else if (responseContent) {
          responseContent.innerHTML = `<p class="error-message">${
            data.detail || "An error occurred while processing your query."
          }</p>`;

          if (response.status === 401 && token) {
            localStorage.removeItem("accessToken");
            if (userName) userName.textContent = "Guest";
            if (logoutBtn) logoutBtn.textContent = "Login";
          }
        }
      } catch (error) {
        console.error("Error:", error);
        if (responseContent) {
          responseContent.innerHTML =
            '<p class="error-message">An error occurred. Please try again later.</p>';
        }
      }
    });
  }

  function formatResponse(text) {
    return text.replace(/\n/g, "<br>");
  }
});
