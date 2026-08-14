import React, { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, Ambulance, ArrowRight, BookOpen, Flame, HeartPulse,
  Mic, Play, Search, ShieldAlert, Siren, Sparkles,
} from 'lucide-react'
import './styles.css'

const VrmAvatar3D = lazy(() => import('./VrmAvatar3D'))
const Avatar3D = lazy(() => import('./Avatar3D'))

const API = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

async function requestJson(path, options = {}, timeoutMs = 15000) {
  const controller = new AbortController()
  const abortFromCaller = () => controller.abort()
  options.signal?.addEventListener('abort', abortFromCaller, { once: true })
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${API}${path}`, { ...options, signal: controller.signal })
    if (!response.ok) {
      const payload = await response.json().catch(() => null)
      throw new Error(payload?.detail || `요청을 처리하지 못했습니다. (${response.status})`)
    }
    return await response.json()
  } catch (error) {
    if (controller.signal.aborted && !options.signal?.aborted) throw new Error('응답 시간이 초과되었습니다. 다시 시도해 주세요.')
    throw error
  } finally {
    window.clearTimeout(timer)
    options.signal?.removeEventListener('abort', abortFromCaller)
  }
}

const PRESETS = [
  { label: '가슴 통증', icon: HeartPulse, text: '가슴이 너무 아파요' },
  { label: '화재', icon: Flame, text: '불이 났어요 계단으로 대피하세요' },
  { label: '병원', icon: Ambulance, text: '병원에 가야 해요' },
  { label: '지진', icon: Activity, text: '지진이 났어요 대피하세요' },
  { label: '감전', icon: Siren, text: '감전됐어요' },
  { label: '위험', icon: ShieldAlert, text: '위험합니다 대피하세요' },
]

const CATEGORY_LABELS = {
  general: '기본 표현', medical: '의료', disaster: '재난', instruction: '안내',
  direction: '방향', accident: '사고',
}

const DEMO_TEXT = '엘리베이터를 이용하지 말고 계단으로 대피하세요'

function AvatarStage({ rendered, loading = false, emptyText = '문장을 변환하면\n아바타 동작이 시작됩니다.', playbackRate = 0.72 }) {
  const [useFallbackAvatar, setUseFallbackAvatar] = useState(false)
  const handleAvatarError = useCallback(() => setUseFallbackAvatar(true), [])
  useEffect(() => setUseFallbackAvatar(false), [rendered])

  return <div className="stage">
    {loading ? <div className="stage-empty"><span className="spinner large"/><span>수어 동작을 준비하고 있습니다.</span></div>
      : rendered ? (rendered.motion?.length
        ? <Suspense fallback={<div className="stage-empty"><span className="spinner large"/><span>3D 아바타를 불러오고 있습니다.</span></div>}>
          {useFallbackAvatar
            ? <Avatar3D motion={rendered.motion} fps={rendered.fps * playbackRate}/>
            : <VrmAvatar3D
              motion={rendered.motion} palmNormals={rendered.palm_normals}
              facialExpressions={rendered.facial_expressions} headRotations={rendered.head_rotations}
              fps={rendered.fps} motionSource={rendered.motion_source}
              playbackRate={playbackRate} onError={handleAvatarError}
            />}
        </Suspense>
        : rendered.url?.endsWith('.mp4')
          ? <video key={rendered.url} src={`${API}${rendered.url}`} aria-label="변환된 수어 아바타 애니메이션" autoPlay loop muted playsInline/>
          : <img src={`${API}${rendered.url}`} alt="변환된 수어 아바타 애니메이션"/>)
        : <div className="stage-empty"><div className="figure"><i/><i/><i/><i/><i/></div><span>{emptyText.split('\n').map((line, index) => <React.Fragment key={line}>{index > 0 && <br/>}{line}</React.Fragment>)}</span></div>}
    <div className="stage-badge"><span/> {useFallbackAvatar ? '3D SKELETON' : rendered?.motion_source?.includes('aihub_landmarks_3d') ? 'AI HUB 3D · 미검수' : rendered?.motion_source?.includes('aihub_') ? 'AI HUB 2D · 미검수' : 'VRM DEMO'} · {rendered?.fps || 20} FPS · {playbackRate.toFixed(2)}×</div>
  </div>
}

function App() {
  const [page, setPage] = useState('converter')
  const [text, setText] = useState(DEMO_TEXT)
  const [result, setResult] = useState(null)
  const [rendered, setRendered] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [datasetStatus, setDatasetStatus] = useState(null)
  const [signs, setSigns] = useState([])
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('all')
  const [selectedSign, setSelectedSign] = useState(null)
  const [dictionaryRendered, setDictionaryRendered] = useState(null)
  const [dictionaryLoading, setDictionaryLoading] = useState(false)
  const [dictionaryError, setDictionaryError] = useState('')
  const [serviceStatus, setServiceStatus] = useState(null)
  const [serviceError, setServiceError] = useState('')
  const [listening, setListening] = useState(false)
  const [playbackRate, setPlaybackRate] = useState(0.72)
  const convertAbortRef = useRef(null)
  const speechRef = useRef(null)

  const loadServiceData = useCallback(async () => {
    setServiceError('')
    try {
      const [status, items] = await Promise.all([
        requestJson('/api/status', {}, 8000),
        requestJson('/api/emergency-signs', {}, 8000),
      ])
      setServiceStatus(status)
      setDatasetStatus(status.dataset)
      setSigns(items.filter(item => item.motion_available && item.motion_source?.startsWith('aihub_')))
      setDictionaryError('')
    } catch (reason) {
      setServiceError(reason.message || 'SignAid 서버에 연결하지 못했습니다.')
      setDictionaryError('등록된 수어 목록을 불러오지 못했습니다.')
    }
  }, [])

  useEffect(() => {
    loadServiceData()
    return () => {
      convertAbortRef.current?.abort()
      speechRef.current?.abort()
    }
  }, [loadServiceData])

  const categories = useMemo(() => ['all', ...new Set(signs.map(sign => sign.category))], [signs])
  const filteredSigns = useMemo(() => {
    const keyword = query.trim().toLocaleLowerCase('ko-KR')
    return signs.filter(sign => {
      const categoryMatch = category === 'all' || sign.category === category
      const searchable = [sign.ko, sign.id, ...(sign.gloss || []), ...(sign.aliases || [])].join(' ').toLocaleLowerCase('ko-KR')
      return categoryMatch && (!keyword || searchable.includes(keyword))
    })
  }, [signs, query, category])

  function changePage(nextPage) {
    setPage(nextPage)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  async function convert(input = text) {
    const value = input.trim()
    if (!value) return
    convertAbortRef.current?.abort()
    const controller = new AbortController()
    convertAbortRef.current = controller
    setLoading(true)
    setError('')
    setRendered(null)
    setResult(null)
    try {
      const parsed = await requestJson('/api/text-to-sign', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: value }), signal: controller.signal,
      })
      setResult(parsed)
      const motion = await requestJson('/api/render', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: value }), signal: controller.signal,
      })
      setRendered(motion)
    } catch (reason) {
      if (reason.name !== 'AbortError') setError(reason.message || '문장을 변환하지 못했습니다.')
    } finally {
      if (convertAbortRef.current === controller) {
        setLoading(false)
        convertAbortRef.current = null
      }
    }
  }

  function usePreset(preset) {
    setText(preset.text)
    convert(preset.text)
  }

  async function playDictionarySign(sign) {
    setSelectedSign(sign)
    setDictionaryRendered(null)
    setDictionaryLoading(true)
    setDictionaryError('')
    try {
      const motion = await requestJson('/api/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: sign.ko, gloss: sign.gloss }),
      })
      setDictionaryRendered(motion)
    } catch (reason) {
      setDictionaryError(`${reason.message} 잠시 후 다시 시도해 주세요.`)
    } finally {
      setDictionaryLoading(false)
    }
  }

  function startVoiceInput() {
    if (listening) {
      speechRef.current?.stop()
      return
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      setError('이 브라우저는 음성 입력을 지원하지 않습니다. Chrome 또는 Edge에서 이용해 주세요.')
      return
    }
    speechRef.current?.abort()
    const recognition = new SpeechRecognition()
    speechRef.current = recognition
    recognition.lang = 'ko-KR'
    recognition.interimResults = true
    recognition.continuous = false
    recognition.onstart = () => { setListening(true); setError('') }
    recognition.onresult = event => {
      const transcript = Array.from(event.results).map(item => item[0].transcript).join(' ').trim()
      if (transcript) setText(transcript)
    }
    recognition.onerror = event => {
      if (event.error !== 'aborted') setError('음성을 인식하지 못했습니다. 마이크 권한을 확인해 주세요.')
    }
    recognition.onend = () => { setListening(false); speechRef.current = null }
    recognition.start()
  }

  return <main>
    <nav className="nav" aria-label="주요 메뉴">
      <button className="brand" type="button" onClick={() => changePage('converter')}><span className="brand-mark"><Sparkles size={18}/></span>SignAid</button>
      <div className="nav-tabs">
        <button type="button" className={page === 'converter' ? 'active' : ''} onClick={() => changePage('converter')}>수어 변환</button>
        <button type="button" className={page === 'dictionary' ? 'active' : ''} onClick={() => changePage('dictionary')}><BookOpen size={15}/> 수어 백과사전</button>
      </div>
      <div className="live"><span/> {datasetStatus?.connected ? `AI Hub ${datasetStatus.samples.toLocaleString('ko-KR')}건 연결` : '오프라인 응급 모드'}</div>
    </nav>

    {serviceError && <div className="service-alert" role="alert">
      <span>서버 연결이 끊겼습니다. {serviceError}</span>
      <button type="button" onClick={loadServiceData}>다시 연결</button>
    </div>}

    {page === 'converter' ? <>
      <section className="hero" id="top">
        <div className="eyebrow">KOREAN SIGN LANGUAGE · EMERGENCY</div>
        <h1>말이 닿지 않는 순간에도<br/><em>의도는 닿아야 하니까</em></h1>
        <p>한국어 응급 문장을 수어 글로스와 사람형 아바타 동작으로 즉시 변환합니다.</p>
      </section>

      <section className="workspace" aria-label="수어 변환기">
        <div className="input-panel">
          <div className="panel-heading"><span>01</span><div><h2>상황을 입력하세요</h2><p>짧고 명확한 문장일수록 더 정확합니다.</p></div></div>
          <label htmlFor="emergency-text" className="sr-only">응급 문장</label>
          <textarea id="emergency-text" value={text} onChange={event => setText(event.target.value)} onKeyDown={event => {
            if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') convert()
          }} maxLength={1000}/>
          <div className="input-footer"><span>{text.length} / 1000 · Ctrl+Enter로 변환</span><button className={`speak ${listening ? 'active' : ''}`} type="button" onClick={startVoiceInput}><Mic size={18}/> {listening ? '입력 완료' : '음성 입력'}</button></div>
          <button className="convert" onClick={() => convert()} disabled={loading || !text.trim()}>
            {loading ? <><span className="spinner"/> 변환하고 있어요</> : <>수어로 변환 <ArrowRight size={20}/></>}
          </button>
          {error && <div className="error" role="alert">{error}</div>}
        </div>

        <div className="result-panel">
          <div className="panel-heading"><span>02</span><div><h2>사람형 수어 아바타</h2><p>{rendered ? '아바타 동작이 재생되고 있습니다.' : '변환 결과가 여기에 표시됩니다.'}</p></div></div>
          <AvatarStage rendered={rendered} loading={loading} playbackRate={playbackRate}/>
          <div className="playback-controls" aria-label="재생 속도">
            <span>재생 속도</span>
            {[0.55, 0.72, 1].map(rate => <button type="button" key={rate} className={playbackRate === rate ? 'active' : ''} onClick={() => setPlaybackRate(rate)}>{rate === 1 ? '보통' : `${rate}×`}</button>)}
          </div>
          <div className="interpretation">
            <div><small>감지된 의도</small><strong>{result?.intent_label || '—'}</strong></div>
            <div className="confidence"><small>신뢰도</small><strong>{result ? `${Math.round(result.confidence * 100)}%` : '—'}</strong></div>
          </div>
          <div className="gloss"><small>수어 글로스</small><div>{result?.gloss?.length ? result.gloss.map((word, index) => <React.Fragment key={`${word}-${index}`}><span>{word}</span>{index < result.gloss.length - 1 && <ArrowRight size={14}/>}</React.Fragment>) : <span className="muted">변환 대기 중</span>}</div></div>
          {result?.reference_samples?.[0] && <div className="reference"><small>AI HUB 유사 문장</small><p>{result.reference_samples[0].text}</p><span>{result.reference_samples[0].category} · 유사도 {Math.round(result.reference_samples[0].similarity * 100)}%</span></div>}
          {rendered?.fallback_motion && <p className="missing">정확히 일치하는 동작이 없어 ‘도움’ 대체 동작을 재생합니다.</p>}
          {result?.missing?.length > 0 && <p className="missing">동작 없음: {result.missing.join(', ')}</p>}
          {rendered && !rendered.expert_validated && <p className="quality-note">{rendered.motion_source?.startsWith('aihub_')
            ? `AI Hub ${rendered.landmark_dimensions}D 주석에서 변환한 동작이며 수어 전문가 검수 전입니다${rendered.tracking_quality ? ` · 추적 품질 ${Math.round(rendered.tracking_quality * 100)}%` : ''}.`
            : '이 결과는 시연용 동작입니다. 긴급 의사결정의 유일한 수단으로 사용하지 마세요.'}</p>}
        </div>
      </section>

      <section className="presets">
        <div><span className="section-index">빠른 선택</span><h2>지금 어떤 도움이 필요한가요?</h2><p>버튼을 누르면 즉시 변환과 재생이 시작됩니다.</p></div>
        <div className="preset-grid">{PRESETS.map(preset => {
          const Icon = preset.icon
          return <button key={preset.label} onClick={() => usePreset(preset)}><span><Icon size={25}/></span><b>{preset.label}</b><ArrowRight size={17}/></button>
        })}</div>
      </section>
    </> : <>
      <section className="dictionary-hero">
        <div>
          <span className="eyebrow">SIGN LIBRARY · {signs.length} ENTRIES</span>
          <h1>손끝으로 찾는<br/><em>수어 백과사전</em></h1>
          <p>AI Hub의 실제 글로스 구간과 키포인트가 연결된 응급 수어만 찾아보고 재생할 수 있습니다.</p>
        </div>
        <div className="dictionary-search"><Search size={19}/><label htmlFor="sign-search" className="sr-only">수어 검색</label><input id="sign-search" value={query} onChange={event => setQuery(event.target.value)} placeholder="단어, 글로스, 비슷한 표현 검색"/></div>
      </section>

      <section className="dictionary-layout" aria-label="수어 백과사전">
        <aside className="dictionary-detail">
          <div className="detail-heading"><span>선택한 수어 · AI Hub 주석 기반 · 전문가 미검수</span>{selectedSign && <b>{CATEGORY_LABELS[selectedSign.category] || selectedSign.category}</b>}</div>
          <AvatarStage rendered={dictionaryRendered} loading={dictionaryLoading} emptyText="오른쪽 목록에서 수어를 선택해 주세요." playbackRate={playbackRate}/>
          {selectedSign ? <div className="sign-meta">
            <span className="sign-id">{selectedSign.id}</span>
            <h2>{selectedSign.ko}</h2>
            <small>수어 글로스</small>
            <div className="meta-gloss">{selectedSign.gloss.map(word => <span key={word}>{word}</span>)}</div>
            {selectedSign.aliases?.length > 0 && <p><b>비슷한 표현</b>{selectedSign.aliases.join(' · ')}</p>}
          </div> : <div className="detail-empty"><BookOpen size={24}/><p>카드를 누르면 이곳에서<br/>수어 동작과 뜻을 확인할 수 있어요.</p></div>}
          {dictionaryError && <div className="error" role="alert">{dictionaryError}</div>}
        </aside>

        <div className="dictionary-browser">
          <div className="browser-heading"><div><span>실제 동작 연결 완료</span><h2>{filteredSigns.length}개의 표현</h2></div><p>카드를 누르면 즉시 재생됩니다.</p></div>
          <div className="category-filters">{categories.map(item => <button key={item} type="button" className={category === item ? 'active' : ''} onClick={() => setCategory(item)}>{item === 'all' ? '전체' : CATEGORY_LABELS[item] || item}</button>)}</div>
          <div className="sign-grid">{filteredSigns.map((sign, index) => <button
            type="button" key={sign.id} className={selectedSign?.id === sign.id ? 'active' : ''}
            onClick={() => playDictionarySign(sign)} disabled={dictionaryLoading}
          >
            <span className="card-number">{String(index + 1).padStart(2, '0')}</span>
            <span className="card-category">{CATEGORY_LABELS[sign.category] || sign.category}</span>
            <h3>{sign.ko}</h3>
            <p>{sign.gloss.join(' · ')}</p>
            <span className="play-icon"><Play size={14} fill="currentColor"/></span>
          </button>)}</div>
          {!filteredSigns.length && <div className="no-results">검색 조건에 맞는 등록 수어가 없습니다.</div>}
        </div>
      </section>
    </>}

    <footer><span>SignAid</span><p>백과사전은 AI Hub 글로스 시간 구간과 실제 키포인트가 연결된 항목만 제공합니다. 미연결 문장은 데모 동작으로 표시됩니다.</p><b>긴급상황은 119</b></footer>
  </main>
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>)
