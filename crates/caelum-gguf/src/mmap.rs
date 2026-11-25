//! Zero-copy read-only memory-mapped file backing (POSIX `mmap` on Unix, owned buffer fallback elsewhere).

#[cfg(unix)]
use caelum_core::CaelumError;
use caelum_core::Result;
use std::ops::Deref;
use std::path::Path;

pub enum MappedFile {
    #[cfg(unix)]
    Mmap {
        ptr: *const u8,
        len: usize,
        _file: std::fs::File,
    },
    Owned(Vec<u8>),
}

unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

#[cfg(unix)]
mod unix_mmap {
    use std::ffi::c_void;

    pub const PROT_READ: i32 = 1;
    pub const MAP_PRIVATE: i32 = 2;
    pub const MADV_DONTNEED: i32 = 4;
    pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, length: usize) -> i32;
        pub fn madvise(addr: *mut c_void, length: usize, advice: i32) -> i32;
    }
}

impl MappedFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let file = std::fs::File::open(path)?;
            let len = file.metadata()?.len() as usize;
            if len == 0 {
                return Ok(Self::Owned(Vec::new()));
            }
            let ptr = unsafe {
                unix_mmap::mmap(
                    std::ptr::null_mut(),
                    len,
                    unix_mmap::PROT_READ,
                    unix_mmap::MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr == unix_mmap::MAP_FAILED {
                return Err(CaelumError::Io(std::io::Error::last_os_error()));
            }
            Ok(Self::Mmap {
                ptr: ptr as *const u8,
                len,
                _file: file,
            })
        }

        #[cfg(not(unix))]
        {
            let bytes = std::fs::read(path)?;
            Ok(Self::Owned(bytes))
        }
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self::Owned(bytes)
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(unix)]
            Self::Mmap { ptr, len, .. } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
            Self::Owned(v) => v.as_slice(),
        }
    }

    /// Advises the OS kernel (`MADV_DONTNEED`) that mapped file pages are no longer needed in
    /// resident system RAM (e.g., after all weights have been uploaded to GPU VRAM).
    /// Future accesses transparently fault in only the specific 4 KB pages touched.
    pub fn advise_dontneed(&self) {
        #[cfg(unix)]
        if let Self::Mmap { ptr, len, .. } = self {
            if *len > 0 && !ptr.is_null() {
                unsafe {
                    let _ = unix_mmap::madvise(
                        *ptr as *mut std::ffi::c_void,
                        *len,
                        unix_mmap::MADV_DONTNEED,
                    );
                }
            }
        }
    }
}

impl Deref for MappedFile {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Self::Mmap { ptr, len, .. } = self {
            if *len > 0 && !ptr.is_null() {
                unsafe {
                    unix_mmap::munmap(*ptr as *mut std::ffi::c_void, *len);
                }
            }
        }
    }
}
