function login() {
  const email = document.querySelector("input[type='email']").value;
  const pass = document.querySelector("input[type='password']").value;

  if(email === "admin@gmail.com" && pass === "1234") {
   window.location.href = "http://localhost:8501"; // your RAG main page
  } else {
    alert("Invalid Credentials!");
  }
}