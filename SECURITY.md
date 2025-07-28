# Security Policy

## Supported Versions

We actively support the following versions of LangForge Documentation:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | ✅ Fully supported |
| 0.9.x   | ⚠️ Security fixes only |
| < 0.9   | ❌ No longer supported |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### 🔒 **For Security Issues**

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please:

1. **Email us directly**: Send details to `security@langforge.dev`
2. **Include details**: Provide a clear description of the vulnerability
3. **Provide reproduction steps**: Help us understand how to reproduce the issue
4. **Suggest fixes**: If you have ideas for fixes, we'd love to hear them

### 📧 **What to Include**

Please include the following information in your security report:

- **Description**: Clear description of the vulnerability
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Impact**: What kind of impact this vulnerability could have
- **Affected versions**: Which versions are affected
- **Suggested fix**: If you have ideas for how to fix it

### ⏱️ **Response Timeline**

- **Initial response**: Within 24 hours
- **Assessment**: Within 3 business days
- **Fix timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next scheduled release

### 🏆 **Recognition**

We believe in recognizing security researchers who help keep our users safe:

- **Public acknowledgment**: With your permission, we'll acknowledge your contribution
- **Hall of fame**: Your name will be added to our security contributors list
- **Swag**: We'll send you some LangForge swag as a thank you

### 🛡️ **Security Best Practices**

When using LangForge Documentation in production:

1. **Keep dependencies updated**: Regularly update all dependencies
2. **Use HTTPS**: Always serve over HTTPS in production
3. **Validate inputs**: Sanitize all user inputs
4. **Monitor logs**: Keep an eye on your application logs
5. **Use environment variables**: Never commit secrets to version control

### 📋 **Security Checklist for Contributors**

Before submitting code:

- [ ] No hardcoded secrets or API keys
- [ ] Input validation for all user inputs
- [ ] Proper error handling that doesn't leak sensitive info
- [ ] Dependencies are up to date
- [ ] Security linting passes
- [ ] Tests include security scenarios

Thank you for helping keep LangForge Documentation secure! 🔒