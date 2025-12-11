# Testing Checklist - Saponify AI Critical Fixes
## Quick Reference for Deployment Verification

**Use this checklist before marking deployment as complete.**

---

## ✅ Pre-Deployment Checks

### Backend (Render)
- [ ] Dependencies installed (`npm install` in `/server`)
- [ ] Environment variables configured:
  - [ ] `GEMINI_API_KEY` set
  - [ ] `GEMINI_MODEL` set to `gemini-2.5-flash`
  - [ ] `NODE_ENV` set to `production`
- [ ] Server starts without errors locally
- [ ] Health endpoint works: `/health`

### Frontend (Local)
- [ ] All new files present in repository
- [ ] No console errors when loading page
- [ ] Calculator initializes successfully
- [ ] Validator initializes successfully
- [ ] DOMPurify loads from CDN

---

## 🧪 Automated Tests

### Unit Tests
- [ ] Navigate to `/tests/test-runner.html`
- [ ] Click "Run All Tests"
- [ ] **All tests must pass** (30+ tests)
- [ ] No errors in console

**Expected Result:**
```
✅ All tests passed!
Total: 30+ tests
```

**If tests fail:**
- ❌ DO NOT DEPLOY
- Review failed test details
- Fix issues before proceeding

---

## 🔐 Security Tests

### Rate Limiting
- [ ] Make 30+ requests rapidly
- [ ] Verify rate limit error appears
- [ ] Wait 1 minute
- [ ] Verify requests work again

**Expected Error:**
```json
{
  "error": "Too many requests from this IP, please try again in a minute.",
  "retryAfter": 60
}
```

### XSS Protection
- [ ] Enter message: `<script>alert('XSS')</script>`
- [ ] Verify no alert pops up
- [ ] Verify message is sanitized in chat
- [ ] Check console for DOMPurify confirmation

**Expected:** No script execution, sanitized HTML shown

### CORS (Production Only)
- [ ] Try to call API from non-victoriarg.com domain
- [ ] Verify CORS error in console

**Expected:** CORS policy blocks unauthorized origins

---

## 🧮 Calculator Tests

### Basic Recipe Calculation
- [ ] Ask: "Create a beginner soap recipe for 500g batch"
- [ ] Verify recipe includes:
  - [ ] Oil amounts in grams and ounces
  - [ ] Lye amount (NaOH)
  - [ ] Water amount
  - [ ] Superfat percentage
  - [ ] **Soap Properties table** (hardness, cleansing, conditioning, etc.)
  - [ ] Safety warnings

**Key Check:** Properties table must be present (this confirms SoapCalculator is being used)

### Recipe with Validation Warnings
- [ ] Create recipe: "500g batch with 400g coconut oil, 100g olive oil"
- [ ] Verify warning appears about high coconut oil (>35%)
- [ ] Warning should say: "⚠️ High coconut oil (>35%) can be drying to skin"

**Expected:** Recipe calculates but shows warnings

### Recipe with Validation Errors
- [ ] Create recipe: "500g batch with 500g coconut oil"
- [ ] Verify error about 100% coconut oil
- [ ] Recipe should still calculate but show strong warnings

**Expected:** Multiple warnings about drying, cleansing values

---

## 🚨 Error Handling Tests

### Network Error
- [ ] Disconnect internet
- [ ] Ask a question
- [ ] Verify specific error message appears
- [ ] Verify fallback to local knowledge base
- [ ] Reconnect internet
- [ ] Verify chat works again

**Expected Message:**
```
Connection Error: Unable to reach the AI server. Using local knowledge base instead.
```

### Timeout Error
- [ ] Ask very complex question requiring long response
- [ ] Wait for timeout (30 seconds)
- [ ] Verify timeout message appears

**Expected Message:**
```
Request Timeout: The AI took too long to respond. Please try asking again with a simpler question.
```

### Invalid Input
- [ ] Enter empty message
- [ ] Click Send
- [ ] Verify nothing happens (input disabled)

**Expected:** No error, input validation prevents empty sends

---

## ♿ Accessibility Tests

### Screen Reader (macOS VoiceOver)
- [ ] Enable VoiceOver (Cmd + F5)
- [ ] Tab to chat input
- [ ] Verify label is announced: "Type your soap making question"
- [ ] Tab to Send button
- [ ] Verify label is announced: "Send message"
- [ ] Ask a question
- [ ] Verify response is announced

**Expected:** All elements have proper labels and are announced

### Keyboard Navigation
- [ ] Tab through all interactive elements
- [ ] Verify focus visible on all elements
- [ ] Press Enter on suggestion buttons
- [ ] Verify suggestion question is sent
- [ ] Press Enter in input field
- [ ] Verify message is sent

**Expected:** Full keyboard navigation works

### Visual Focus Indicators
- [ ] Tab through elements
- [ ] Verify visible focus ring on each element
- [ ] Verify contrast is sufficient

**Expected:** Clear focus indicators on all elements

---

## 📱 Mobile Tests

### Responsive Design (iPhone/Android)
- [ ] Open on mobile device
- [ ] Verify chat fits screen
- [ ] Verify input is accessible
- [ ] Verify suggestion buttons wrap properly
- [ ] Verify messages are readable
- [ ] Test landscape orientation

**Expected:** Fully functional on mobile

### Touch Interactions
- [ ] Tap suggestion buttons
- [ ] Tap Send button
- [ ] Scroll chat messages
- [ ] Type in input field

**Expected:** All touch interactions work smoothly

---

## 🔄 Integration Tests

### Full Recipe Creation Flow
1. [ ] User clicks "Calculate Recipe 🧮" suggestion
2. [ ] AI asks for batch size
3. [ ] User enters "500g"
4. [ ] AI asks for oils
5. [ ] User enters "300g olive oil, 150g coconut oil, 50g castor oil"
6. [ ] AI asks about superfat
7. [ ] User enters "calculate"
8. [ ] Recipe appears with:
   - [ ] Complete ingredient list
   - [ ] Soap properties table
   - [ ] Validation results (if any)
   - [ ] Safety warnings

**Expected:** Complete flow works end-to-end with accurate calculations

### Error Recovery
1. [ ] Start recipe calculation
2. [ ] Disconnect internet midway
3. [ ] Verify error message
4. [ ] Reconnect internet
5. [ ] Continue conversation
6. [ ] Verify chat recovers

**Expected:** Graceful error handling, conversation continues

---

## 📊 Performance Tests

### Page Load
- [ ] Hard refresh page (Cmd + Shift + R)
- [ ] Measure load time
- [ ] Verify < 3 seconds on fast connection

**Expected:** Fast initial load

### Response Time
- [ ] Ask simple question
- [ ] Measure time to response
- [ ] Verify < 2 seconds with AI
- [ ] Verify instant with local knowledge

**Expected:** Reasonable response times

### Memory Leaks
- [ ] Have 10+ message conversation
- [ ] Check browser memory usage
- [ ] Verify no excessive growth

**Expected:** Stable memory usage

---

## 🔍 Code Quality Checks

### Browser Console
- [ ] Open developer tools
- [ ] Check console for errors
- [ ] Verify only expected logs appear:
  - [ ] "✅ SoapCalculator initialized successfully"
  - [ ] "✅ RecipeValidator initialized successfully"

**Expected:** No errors, only success logs

### Network Tab
- [ ] Open network tab
- [ ] Ask a question
- [ ] Verify API request succeeds
- [ ] Check response time
- [ ] Verify payload size reasonable

**Expected:** Successful API calls, reasonable sizes

---

## 🎯 User Experience Tests

### First-Time User
- [ ] Clear browser cache
- [ ] Load page fresh
- [ ] Verify welcome message is clear
- [ ] Try suggestion buttons
- [ ] Calculate a recipe
- [ ] Verify experience is intuitive

**Expected:** New users can immediately calculate a recipe

### Error States
- [ ] Trigger each error type
- [ ] Verify error messages are clear
- [ ] Verify recovery instructions work
- [ ] Verify chat never becomes unusable

**Expected:** Users always know what to do next

---

## ✅ Deployment Verification

After deploying to production:

### Backend (Render)
- [ ] Visit: `https://saponify-ai-backend.onrender.com/health`
- [ ] Verify returns: `{"status":"ok","message":"Saponify AI proxy server is running"}`
- [ ] Check Render logs for no errors
- [ ] Verify environment variables are set

### Frontend (GitHub Pages)
- [ ] Visit: `https://victoriarg.com/saponifyai.html`
- [ ] Clear cache, hard refresh
- [ ] Verify page loads
- [ ] Check console for initialization logs
- [ ] Test recipe calculation
- [ ] Verify validation warnings appear

### Test Suite
- [ ] Visit: `https://victoriarg.com/tests/test-runner.html`
- [ ] Click "Run All Tests"
- [ ] Verify all tests pass

---

## 🚨 Critical Issues That Block Deployment

**DO NOT DEPLOY if any of these are true:**

- ❌ Unit tests failing
- ❌ SoapCalculator not initializing
- ❌ Recipe calculations missing properties
- ❌ XSS protection not working
- ❌ Rate limiting not enforcing
- ❌ Backend health check failing
- ❌ Console errors on page load
- ❌ Chat completely broken
- ❌ Validation not showing warnings

**If any critical issues:** Fix before deploying!

---

## ✅ Sign-Off

After completing all tests above:

**Tested by:** _________________

**Date:** _________________

**Environment:**
- [ ] Local development
- [ ] Staging
- [ ] Production

**Test Results:**
- Total tests run: _____
- Tests passed: _____
- Tests failed: _____

**Critical issues found:** _____

**Approved for deployment:** [ ] Yes [ ] No

**Notes:**
_______________________________________________________
_______________________________________________________
_______________________________________________________

---

## 📋 Quick Reference: Expected Console Logs

When page loads successfully, console should show:

```
✅ SoapCalculator initialized successfully
✅ RecipeValidator initialized successfully
SoapCalculator tests loaded. Run tests with: runner.run()
```

**No errors should appear in console unless intentionally triggered.**

---

## 🎉 Success Criteria

**Deployment is successful when:**

1. ✅ All unit tests pass (30+)
2. ✅ Backend health check returns OK
3. ✅ Chat responds to questions
4. ✅ Recipe calculations include properties
5. ✅ Validation warnings appear appropriately
6. ✅ Error messages are specific
7. ✅ Accessibility works
8. ✅ Mobile experience is smooth
9. ✅ No console errors
10. ✅ Rate limiting enforces

**If all criteria met:** ✅ Deployment successful!

**If any criteria fail:** ❌ Investigate and fix before marking complete.

---

*Use this checklist every time you deploy changes to Saponify AI.*
