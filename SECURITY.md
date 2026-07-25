# Security Policy

## 🔒 Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.1.x   | :white_check_mark: |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## 🐛 Reporting a Vulnerability

We take the security of DeepRAG seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please Do NOT:

- Open a public GitHub issue
- Disclose the vulnerability publicly before it has been addressed

### Please Do:

1. **Email us directly** at: [your-email@example.com]
2. **Include the following information**:
   - Type of vulnerability
   - Full paths of source file(s) related to the vulnerability
   - Location of the affected source code (tag/branch/commit or direct URL)
   - Step-by-step instructions to reproduce the issue
   - Proof-of-concept or exploit code (if possible)
   - Impact of the vulnerability
   - Suggested fix (if available)

### What to Expect:

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
- **Assessment**: We will assess the vulnerability and determine its impact and severity
- **Fix**: We will work on a fix and release it as soon as possible
- **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

### Timeline:

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Release**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next regular release

## 🛡️ Security Best Practices

When using DeepRAG in production:

### 1. API Keys and Secrets

- **Never commit** API keys or secrets to version control
- Use environment variables or secret management systems (e.g., AWS Secrets Manager, HashiCorp Vault)
- Rotate API keys regularly
- Use different keys for development, staging, and production

### 2. Network Security

- Use HTTPS/TLS for all external communications
- Implement rate limiting to prevent abuse
- Use firewall rules to restrict access
- Enable CORS only for trusted domains

### 3. Input Validation

- Validate and sanitize all user inputs
- Implement query length limits
- Use parameterized queries for database operations
- Escape special characters in user-provided content

### 4. Authentication & Authorization

- Implement authentication for production deployments
- Use OAuth 2.0 or JWT for API authentication
- Implement role-based access control (RBAC)
- Log all authentication attempts

### 5. Dependency Management

- Keep dependencies up to date
- Use Dependabot for automated security updates
- Regularly audit dependencies with `pip-audit`
- Pin dependency versions in production

### 6. Monitoring & Logging

- Enable comprehensive logging
- Monitor for suspicious activity
- Set up alerts for security events
- Regularly review logs

### 7. Container Security

- Use official base images
- Scan images for vulnerabilities
- Run containers as non-root user
- Keep images updated

## 🔍 Security Scanning

We use the following tools to maintain security:

- **Dependabot**: Automated dependency updates
- **GitHub Security Advisories**: Vulnerability alerts
- **Snyk**: Continuous security monitoring (optional)
- **Bandit**: Python security linter

### Run Security Checks Locally:

```bash
# Install security tools
pip install bandit safety pip-audit

# Run Bandit (Python security linter)
bandit -r src/ -ll

# Check for known vulnerabilities
safety check

# Audit dependencies
pip-audit
```

## 📚 Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)

## 🙏 Acknowledgments

We would like to thank the following security researchers for responsibly disclosing vulnerabilities:

<!-- List will be updated as vulnerabilities are reported and fixed -->

---

**Last Updated**: 2026-05-27
