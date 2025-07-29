module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true,
    jest: true
  },
  extends: [
    'eslint:recommended'
  ],
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module'
  },
  plugins: [
    'security',
    'markdown'
  ],
  rules: {
    'indent': ['error', 2],
    'quotes': ['error', 'single'],
    'semi': ['error', 'always'],
    'no-unused-vars': 'error',
    'no-console': 'warn',
    'security/detect-object-injection': 'error',
    'security/detect-non-literal-fs-filename': 'warn',
    'security/detect-eval-with-expression': 'error'
  },
  overrides: [
    {
      files: ['**/*.md'],
      processor: 'markdown/markdown'
    },
    {
      files: ['**/*.md/*.js'],
      rules: {
        'no-undef': 'off',
        'no-unused-vars': 'off'
      }
    },
    {
      // Relax rules for generated bundle files
      files: ['**/dist/**/*.js'],
      rules: {
        'no-console': 'off',
        'no-unused-vars': 'off',
        'quotes': 'off',
        'indent': 'off',
        'security/detect-object-injection': 'off'
      }
    }
  ]
};