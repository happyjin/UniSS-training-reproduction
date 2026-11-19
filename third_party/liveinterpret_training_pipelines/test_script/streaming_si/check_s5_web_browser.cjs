/* Browser checks against the unchanged original S5 HTML and real GPU backend.
 * NODE_PATH=/tmp/s5-browser-tools/node_modules node this.cjs URL AUDIO.wav OUTDIR [zh|en] [upload|microphone]
 * S5_TEST_SELF_SIGNED_TLS=1 accepts the development certificate in this test only.
 */
const fs = require('fs');
const path = require('path');
const {chromium} = require('playwright');

async function main() {
  const [url, audio, out, lang='zh', mode='upload'] = process.argv.slice(2);
  if (!url || !audio || !out || !['upload','microphone'].includes(mode)) throw Error('URL AUDIO OUT [LANG] [upload|microphone] required');
  fs.mkdirSync(out, {recursive:false});
  const options = {headless:true, args:['--no-sandbox','--autoplay-policy=no-user-gesture-required']};
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE) options.executablePath=process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
  if (mode==='microphone') options.args.push('--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream','--use-file-for-fake-audio-capture='+path.resolve(audio));
  const browser=await chromium.launch(options);
  const context=await browser.newContext({acceptDownloads:true, ignoreHTTPSErrors:process.env.S5_TEST_SELF_SIGNED_TLS==='1', permissions:mode==='microphone'?['microphone']:[], viewport:{width:1100,height:1000}});
  const page=await context.newPage();
  const errors=[], dialogs=[], wire=[], sent=[];
  let config, response;
  page.on('pageerror',e=>errors.push(e.message));
  page.on('dialog',async d=>{dialogs.push(d.message());await d.dismiss();});
  page.on('websocket',socket=>{
    socket.on('framereceived',e=>{try{wire.push(JSON.parse(e.payload.toString()));}catch(err){errors.push(String(err));}});
    socket.on('framesent',e=>{if(typeof e.payload==='string'){try{sent.push(JSON.parse(e.payload));}catch(_){}}});
    socket.on('socketerror',e=>errors.push(String(e)));
  });
  const began=Date.now();
  try {
    await page.goto(url,{waitUntil:'networkidle',timeout:120000});
    await page.waitForFunction(()=>document.querySelector('#langSel').options.length>0,null,{timeout:120000});
    config=await page.evaluate(()=>CFG);
    if(!config.model_dir || !config.adapter) throw Error('missing actual checkpoint configuration');
    await page.locator('#langSel').selectOption(lang);
    if(process.env.S5_TEST_CHUNK_MS) await page.locator('#chunkSel').selectOption(process.env.S5_TEST_CHUNK_MS);
    if(process.env.S5_TEST_AFTER_WRITE_MS!==undefined) await page.locator('#afterChunkSel').selectOption(process.env.S5_TEST_AFTER_WRITE_MS==='none'?'':process.env.S5_TEST_AFTER_WRITE_MS);
    const selected=await page.evaluate(()=>typeof chunkSettings==='function'?chunkSettings():null);
    if(mode==='upload'){
      const pending=page.waitForResponse(r=>new URL(r.url()).pathname==='/translate' && r.request().method()==='POST',{timeout:240000});
      await page.locator('#fileIn').setInputFiles(audio);
      response=await pending;
      if(response.status()!==200 || !response.headers()['content-type'].includes('text/event-stream')) throw Error('original upload SSE protocol unavailable');
      const body=await response.text();
      for(const part of body.split('\n\n')) if(part.startsWith('data: ')) wire.push(JSON.parse(part.slice(6)));
      await page.waitForFunction(()=>!busy && !!document.querySelector('.player'),null,{timeout:240000});
    }else{
      await page.locator('#recBtn').click();
      await page.waitForFunction(()=>live,null,{timeout:120000});
      await page.waitForTimeout(13000);
      await page.locator('#recBtn').click();
      await page.waitForFunction(()=>!live && !!document.querySelector('.player'),null,{timeout:120000});
      if(!sent.some(e=>e.type==='start') || !sent.some(e=>e.type==='stop') || !sent.some(e=>e.type==='end')) throw Error('original microphone handshake/finish incomplete');
    }
    if(errors.length || dialogs.length) throw Error(JSON.stringify({errors,dialogs}));
    if(await page.locator('.bubble.err').count()) throw Error('page displayed inference failure');
    const done=wire.findLast(e=>e.type==='done');
    const chunks=wire.filter(e=>e.type==='audio');
    if(!done || !chunks.length || wire.some(e=>e.type==='error')) throw Error('missing final metrics or actual PCM');
    if(selected && (done.metrics.chunk_ms!==selected.chunk_ms || done.metrics.chunk_ms_after_write!==selected.chunk_ms_after_write)) throw Error('backend did not apply selected chunk configuration');
    if(selected && await page.evaluate(()=>document.querySelector('#chunkSel').disabled || document.querySelector('#afterChunkSel').disabled)) throw Error('chunk controls did not unlock after completion');
    const decisions=wire.filter(e=>e.type==='decision');
    const firstWrite=decisions.findIndex(e=>e.action==='write');
    const needsSwitch=selected && selected.chunk_ms_after_write && selected.chunk_ms_after_write!==selected.chunk_ms;
    const observedSwitch=needsSwitch && firstWrite>=0 && decisions.slice(0,firstWrite+1).some(e=>e.n_audio_frames===selected.chunk_ms/80) && decisions.slice(firstWrite+1).some(e=>e.n_audio_frames===selected.chunk_ms_after_write/80);
    if(needsSwitch && !observedSwitch) throw Error('selected source-frame transition was not observed');
    if((await page.locator('.msg:not(.user) .bubble').last().textContent())!==done.text) throw Error('rendered translation differs from final text');
    const download=page.waitForEvent('download');
    await page.locator('.player a.dl').last().click();
    await (await download).saveAs(path.join(out,'s5-web-output.wav'));
    const wav=fs.readFileSync(path.join(out,'s5-web-output.wav'));
    const expected=Buffer.concat(chunks.map(e=>Buffer.from(e.pcm,'base64')));
    if(wav.toString('ascii',0,4)!=='RIFF' || wav.readUInt32LE(24)!==24000 || !wav.subarray(44).equals(expected)) throw Error('download differs from actual emitted PCM');
    if(Math.abs(expected.length/2/24000-done.output_sec)>1/24000) throw Error('WAV duration differs from backend');
    await page.screenshot({path:path.join(out,'s5-web.png'),fullPage:true});
    fs.writeFileSync(path.join(out,'events.jsonl'),wire.map(e=>JSON.stringify(e.type==='audio'?{...e,pcm:undefined,pcm_bytes:Buffer.from(e.pcm,'base64').length}:e)).join('\n')+'\n');
    fs.writeFileSync(path.join(out,'browser-report.json'),JSON.stringify({passed:true,original_page:true,actual_gpu_server:true,config,selected_chunks:selected,chunk_switch_observed:observedSwitch,transport_mode:mode,synthetic_microphone:mode==='microphone',human_listening_verified:false,pcm_chunks:chunks.length,wav_samples:expected.length/2,text:done.text,metrics:done.metrics,browser_wall_sec:(Date.now()-began)/1000,errors,dialogs},null,2));
  } catch(e) {
    await page.screenshot({path:path.join(out,'failure.png'),fullPage:true}).catch(()=>{});
    fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({error:String(e),errors,dialogs,config,sent,events:wire.map(e=>e.type==='audio'?{...e,pcm:undefined}:e)},null,2));
    throw e;
  } finally {await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
