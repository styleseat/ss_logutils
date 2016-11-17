# ss\_logutils: Python logging utilities

This library extends the python logging package, providing additional handlers
suitable for production use.

# Sample Usage

```python
# Create a fork-safe handler which child processes can safely inherit and
# use without corrupting the write stream. The handler rotates the log
# file every 256MB or 3600 seconds (one hour), whichever comes first.
import ss_logutils.handlers.ForkSafeArchivingFileHandler
logger = logging.getLogger('')
handler = new ForkSafeArchivingFileHandler(
  '/var/log/my_app/current/safe.log',
  '/var/log/my_app/archived',
  maxSize=256 * 2**20,
  interval=3600)
logger.addHandler(handler)
```
