# 📄 AI Resume Screener

An intelligent resume screening application powered by AI that analyzes, evaluates, and provides detailed feedback on resumes. This tool helps both candidates and recruiters streamline the resume review process using advanced natural language processing.

## 🎯 Features

- **Resume Analysis**: AI-powered analysis of resume content, formatting, and structure
- **Candidate Feedback**: Detailed insights and suggestions for resume improvement
- **Recruiter Dashboard**: Tools for recruiters to screen and manage candidates
- **User Authentication**: Secure login system for candidates and recruiters
- **Multi-role Support**: Separate experiences for candidates and recruiting teams
- **Session Management**: Track recruiter screening sessions and candidate records
- **Notes & Comments**: Add and manage notes on candidate profiles
- **Data Persistence**: Secure storage of user data and screening records

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Streamlit
- Additional dependencies listed in requirements (if available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chinnu48/AI_Resume_Screener.git
   cd AI_Resume_Screener
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

## 📋 User Roles

### Candidate
- Upload and analyze resumes
- Receive AI-powered feedback on resume quality
- Get suggestions for improvement
- View screening results

### Recruiter
- Screen multiple candidate resumes
- Track screening sessions
- Add notes and comments on candidates
- Manage candidate database

## 🔐 Authentication

The application includes a secure authentication system:
- User registration for candidates and recruiters
- Password hashing using SHA-256
- Session-based authentication
- Role-based access control

### Default Users
The application comes with default user accounts for testing purposes.

## 📁 Project Structure

- **app.py** - Main application file containing:
  - Authentication system (login, registration, logout)
  - Resume analysis functionality
  - User management and data persistence
  - Streamlit UI components
  - Database operations

## 💾 Data Storage

User data and screening records are stored locally:
- User credentials and profiles
- Candidate screening records
- Recruiter session data
- Messages and notes

## 🛠️ Technology Stack

- **Python 3** - Core programming language
- **Streamlit** - Web framework for building the UI
- **AI/NLP** - For intelligent resume analysis
- **JSON** - Data persistence format

## 🔄 Workflow

### For Candidates:
1. Register or login to your account
2. Upload your resume
3. Get AI-powered analysis and feedback
4. View improvement suggestions
5. Update your resume based on feedback

### For Recruiters:
1. Login to recruiter account
2. Access candidate pool
3. Review and screen resumes
4. Add notes and comments
5. Track screening progress

## ⚙️ Configuration

Key files and configurations:
- User data: `users.json`
- Application data: `data.json`
- Default users: Configured in `DEFAULT_USERS` dictionary

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📝 License

This project is currently unlicensed. Please see the repository for more details.

## 👤 Author

**chinnu48**
- GitHub: [@chinnu48](https://github.com/chinnu48)
- Repository: [AI_Resume_Screener](https://github.com/chinnu48/AI_Resume_Screener)

## 🐛 Issues & Feedback

If you encounter any issues or have suggestions for improvements, please:
- Open an issue on GitHub
- Provide detailed description of the problem
- Include steps to reproduce (if applicable)

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Python Documentation](https://docs.python.org/3/)

## 🔮 Future Enhancements

Potential features for future versions:
- Advanced AI scoring system
- Resume templates and samples
- Integration with job posting platforms
- Export reports in multiple formats
- Advanced analytics dashboard
- Email notifications
- API endpoints for integration

---

**Last Updated**: November 2025
