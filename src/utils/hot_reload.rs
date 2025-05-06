//! 핫 리로드 유틸리티
//! 개발 중 코드 변경 시 자동으로 라이브러리를 다시 로드

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;
use std::time::SystemTime;

/// 파일 모니터 구조체
pub struct FileMonitor {
    /// 감시할 파일 경로와 마지막 수정 시간
    files: Arc<RwLock<HashMap<PathBuf, SystemTime>>>,
    /// 콜백 함수들
    callbacks: Arc<Mutex<Vec<Box<dyn Fn() + Send + 'static>>>>,
    /// 모니터링 활성화 상태
    active: Arc<RwLock<bool>>,
}

impl FileMonitor {
    /// 새 파일 모니터 생성
    pub fn new() -> Self {
        FileMonitor {
            files: Arc::new(RwLock::new(HashMap::new())),
            callbacks: Arc::new(Mutex::new(Vec::new())),
            active: Arc::new(RwLock::new(false)),
        }
    }

    /// 파일 추가
    pub fn add_file<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let path = path.as_ref().to_path_buf();
        let metadata = fs::metadata(&path)?;
        let modified = metadata.modified()?;

        let mut files = self.files.write().unwrap();
        files.insert(path, modified);

        Ok(())
    }

    /// 콜백 추가
    pub fn add_callback<F>(&self, callback: F)
    where
        F: Fn() + Send + 'static,
    {
        let mut callbacks = self.callbacks.lock().unwrap();
        callbacks.push(Box::new(callback));
    }

    /// 모니터링 시작
    pub fn start(&self, interval_ms: u64) {
        let files = Arc::clone(&self.files);
        let callbacks = Arc::clone(&self.callbacks);
        let active = Arc::clone(&self.active);

        {
            let mut active_guard = active.write().unwrap();
            *active_guard = true;
        }

        thread::spawn(move || {
            while *active.read().unwrap() {
                let mut changed = false;

                // 파일 변경 확인
                {
                    let mut files_guard = files.write().unwrap();
                    for (path, last_modified) in files_guard.iter_mut() {
                        if let Ok(metadata) = fs::metadata(path) {
                            if let Ok(current_modified) = metadata.modified() {
                                if current_modified > *last_modified {
                                    *last_modified = current_modified;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                // 변경이 감지되면 콜백 호출
                if changed {
                    let callbacks_guard = callbacks.lock().unwrap();
                    for callback in callbacks_guard.iter() {
                        callback();
                    }
                }

                thread::sleep(Duration::from_millis(interval_ms));
            }
        });
    }

    /// 모니터링 중지
    pub fn stop(&self) {
        let mut active = self.active.write().unwrap();
        *active = false;
    }
}

impl Drop for FileMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}
